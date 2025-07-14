FILE=local/token.env
TOK=`cat $(FILE)`

.PHONY: build deploy test prof docs pages

build:
	python setup.py bdist_wheel sdist
	twine check dist/*

deploy:
	twine upload dist/* -u plutoniumm -p $(TOK)
	rm -rf build dist qudit.egg-info

test:
	pip install .
	python test.py

prof:
	cd tests && python -m cProfile -o program.prof bench_GHZ.py && snakeviz program.prof;

docs:
	cd docs && make html;
	cd docs/_build/html && php -S localhost:3001 & open http://localhost:3001;

pages:
	cd docs && make html;
	touch docs/_build/html/.nojekyll;
	npx gh-pages -d docs/_build/html -t
