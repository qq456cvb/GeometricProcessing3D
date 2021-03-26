
# Python interface
setup(
    name="MinkowskiEngine",
    install_requires=["torch", "numpy"],
    packages=["MinkowskiEngine", "MinkowskiEngine.utils", "MinkowskiEngine.modules"],
    package_dir={"MinkowskiEngine": "./MinkowskiEngine"},
    ext_modules=ext_modules,
    include_dirs=[str(SRC_PATH), str(SRC_PATH / "3rdparty"), *include_dirs],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.6",
)