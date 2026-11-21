import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


class BuildLittleBit(BuildExtension):
    def run(self) -> None:
        root = Path(__file__).resolve().parent
        build = Path(self.build_temp).resolve() / "littlebit"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DBUILD_TESTING=OFF",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build), "--target", "littlebit"], check=True)
        self.extensions[0].extra_objects = [str(build / "liblittlebit.a")]
        super().run()


setup(
    ext_modules=[
        CppExtension(
            "littlebit._native",
            ["src/torch.cpp"],
            include_dirs=[str(Path(__file__).resolve().parent / "include")],
            extra_compile_args=["-O2", "-g0"],
            extra_link_args=["-pthread"],
        )
    ],
    cmdclass={"build_ext": BuildLittleBit},
)
