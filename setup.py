from setuptools import setup, find_packages

setup(name='booknlp-plus', 
	version='1.0.7', 
	packages=find_packages(),
	py_modules=['booknlp'],
	url="https://github.com/DrewThomasson/booknlp/tree/json-patch-1",
	author="David Bamman (Original), Drew Thomasson (Fork)",
	author_email="dbamman@berkeley.edu",
	description="Enhanced fork of BookNLP with JSON patch support and additional features",
	long_description="""
	BookNLP Plus: An Enhanced Fork of BookNLP
	
	This is an enhanced fork of the original BookNLP by David Bamman, featuring:
	- JSON patch support
	- Sentence transformers integration
	- Updated dependencies for Python 3.9-3.12
	- Additional improvements and bug fixes
	
	Original repository: https://github.com/dbamman/book-nlp
	Original author: David Bamman (UC Berkeley)
	
	Enhanced fork repository: https://github.com/DrewThomasson/booknlp/tree/json-patch-1
	Fork maintainer: Drew Thomasson
	""",
	long_description_content_type="text/plain",
	include_package_data=True, 
	license="MIT",
	python_requires='>=3.9,<3.13',
	install_requires=[
		'torch>=2.0.0',
		'tokenizers>=0.13.0',
		'spacy>=3.5.0',
		'transformers>=4.30.0',
		'sentence-transformers',
		'tf-keras',
		'numpy>=1.24.0',
		'tqdm>=4.65.0',
		'filelock>=3.12.0',
		'regex>=2023.8.8',
		'requests>=2.31.0',
		'pyyaml>=6.0.1',
		'packaging>=23.0',
		'pandas>=1.3.0',
		'matplotlib>=3.4.0',
		'networkx>=2.6.0',
		'jinja2>=3.0.0'
	],
	project_urls={
		"Original Repository": "https://github.com/dbamman/book-nlp",
		"Fork Repository": "https://github.com/DrewThomasson/booknlp",
		"Current Branch": "https://github.com/DrewThomasson/booknlp/tree/json-patch-1",
		"Bug Reports": "https://github.com/DrewThomasson/booknlp/issues",
	},
	classifiers=[
		"Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Text Processing :: Linguistic",
	],
	keywords="booknlp, nlp, natural language processing, literature, fiction, character analysis, entity recognition, sentence transformers, plus",
)