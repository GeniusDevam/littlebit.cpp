#include "littlebit.h"

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <memory>
#include <optional>
#include <vector>

namespace {

using at::Tensor;

void check_tensor(const Tensor &t, at::ScalarType dtype, at::IntArrayRef shape) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.sizes() == shape &&
                    t.is_contiguous(),
                "Expected a contiguous CPU tensor of dtype ", dtype, " and shape ", shape);
}

class Layer : public torch::CustomClassHolder {
  public:
    Layer(int64_t in, int64_t out, int64_t rank, const std::vector<Tensor> &weights,
          const std::optional<Tensor> &bias)
        : in_(in), out_(out), layer_(nullptr, lb_layer_free) {
        TORCH_CHECK(in > 0 && out > 0 && rank > 0 && in <= INT32_MAX && out <= INT32_MAX &&
                        rank <= INT32_MAX,
                    "Invalid LittleBit dimensions");
        TORCH_CHECK(weights.size() == 5 || weights.size() == 10,
                    "Expected one or two paths with five tensors each");
        lb_path paths[2];
        for (size_t p = 0; p < weights.size() / 5; ++p) {
            const Tensor *w = weights.data() + p * 5;
            check_tensor(w[0], at::kInt, {out, (rank + 31) / 32});
            check_tensor(w[1], at::kInt, {rank, (in + 31) / 32});
            check_tensor(w[2], at::kFloat, {out});
            check_tensor(w[3], at::kFloat, {rank});
            check_tensor(w[4], at::kFloat, {in});
            paths[p] = {reinterpret_cast<const uint32_t *>(w[0].const_data_ptr<int32_t>()),
                        reinterpret_cast<const uint32_t *>(w[1].const_data_ptr<int32_t>()),
                        w[2].const_data_ptr<float>(), w[3].const_data_ptr<float>(),
                        w[4].const_data_ptr<float>()};
        }
        if (bias) {
            check_tensor(*bias, at::kFloat, {out});
        }
        layer_.reset(lb_layer_create(in, out, rank, paths,
                                     weights.size() == 10 ? paths + 1 : nullptr,
                                     bias ? bias->const_data_ptr<float>() : nullptr));
        TORCH_CHECK(layer_,
                    "Could not create LittleBit layer (invalid padding or allocation failure)");
    }

    Tensor forward(const Tensor &x, int64_t kernel, int64_t threads) const {
        TORCH_CHECK(x.device().is_cpu() && x.dim() >= 1 && x.size(-1) == in_,
                    "Expected CPU activations with last dimension ", in_);
        TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                        x.scalar_type() == at::kBFloat16,
                    "Expected float32, float16 or bfloat16 activations");
        TORCH_CHECK(!at::GradMode::is_enabled() || !x.requires_grad(),
                    "LittleBit is inference only");
        TORCH_CHECK(kernel >= LB_AUTO && kernel <= LB_BATCHED && threads >= 0 && threads <= 32,
                    "Invalid kernel or thread count");
        auto shape = x.sizes().vec();
        shape.back() = out_;
        Tensor y = at::empty(shape, x.options().dtype(at::kFloat));
        if (y.numel()) {
            Tensor input = x.to(at::kFloat).contiguous();
            const lb_status status =
                lb_layer_forward_ex(layer_.get(), input.const_data_ptr<float>(), x.numel() / in_,
                                    y.data_ptr<float>(), static_cast<lb_kernel>(kernel), threads);
            TORCH_CHECK(status == LB_OK, "LittleBit forward failed with status ", status);
        }
        return y.to(x.scalar_type());
    }

  private:
    int64_t in_;
    int64_t out_;
    std::unique_ptr<lb_layer, decltype(&lb_layer_free)> layer_;
};

Tensor forward(const Tensor &x, const c10::intrusive_ptr<Layer> &layer, int64_t kernel,
               int64_t threads) {
    return layer->forward(x, kernel, threads);
}

bool kernel_available(int64_t kernel) {
    return kernel >= LB_AUTO && kernel <= LB_BATCHED &&
           lb_kernel_available(static_cast<lb_kernel>(kernel));
}

} // namespace

TORCH_LIBRARY(littlebit, m) {
    m.class_<Layer>("Layer").def(torch::init<int64_t, int64_t, int64_t, const std::vector<Tensor> &,
                                             const std::optional<Tensor> &>());
    m.def("forward(Tensor x, __torch__.torch.classes.littlebit.Layer layer, int kernel, int "
          "threads) -> Tensor");
    m.def("kernel_available", kernel_available);
}

TORCH_LIBRARY_IMPL(littlebit, CPU, m) { m.impl("forward", forward); }
