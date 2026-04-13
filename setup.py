from setuptools import find_packages, setup


setup(
    name="mare-retrieval",
    version="0.2.0",
    description="Evidence-first PDF retrieval library that returns the best page, exact snippet, and visual evidence for a query.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=["pypdf>=4.0", "pypdfium2>=4.30.0"],
    extras_require={
        "dev": ["pytest>=8.0"],
        "ui": ["streamlit>=1.12,<2.0", "altair<5"],
    },
    entry_points={
        "console_scripts": [
            "mare-demo=mare.demo:main",
            "mare-ingest=mare.ingest:main",
            "mare-ask=mare.ask:main",
            "mare-ui=mare.streamlit_app:main",
        ]
    },
)
