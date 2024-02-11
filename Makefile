install:
		pip install --upgrade pip &&\ pip install -r requirements.txt

lint:
		pylint --disable=R,C,E1120 linear.py

format:
		black *.py

all: install lint test