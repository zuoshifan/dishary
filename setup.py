from setuptools import setup, find_packages


setup(
    name = 'dishary',
    version = 0.1,

    packages = find_packages(),
    scripts=['scripts/tlmdl_vis', 'scripts/tlmk_uv', 'scripts/tlpol_rot', 'scripts/tlmk_img', 'scripts/tlcl_img', 'scripts/tlplt_img', 'scripts/tlplt_uv'],
    requires = ['numpy', 'aipy'],  # Probably should change this.

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@ba.ac.cn",
    description = "Data processing package for Tianlai dish array.",
    license = "GPL v3.0",
    url = "https://github.com/zuoshifan/dishary"
)
