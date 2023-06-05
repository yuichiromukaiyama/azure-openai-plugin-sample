SHELL := /bin/bash

install:
		pip install -r install.txt

update-deps:
		pip freeze > install.txt
	