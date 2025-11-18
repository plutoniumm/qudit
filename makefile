FILE=local/token.env
TOK=`cat $(FILE)`

.PHONY: build deploy test prof docs pages

build:
	python setup.py bdist_wheel sdist
	twine check dist/*

deploy:
	twine upload dist/* -u __token__ -p $(TOK)
	rm -rf build dist qudit.egg-info

test:
	cd tests && python algo.py
	cd tests && python ECC.py
	cd tests && python circuit_M.py
	cd tests && python circuit_V.py
	cd tests && python gates.py
	cd tests && python gd.py
	cd tests && python metrics.py
	cd tests && python qsvt.py
	cd tests && python primitives.py


prof:
	cd tests && python -m cProfile -o program.prof bench_fast.py && snakeviz program.prof;

docs:
	cd docs && make html;
	cd docs/_static && openssl rand -base64 5 > rand;

pages:
	cd docs && make html;
	touch docs/_build/html/.nojekyll;
	npx gh-pages -d docs/_build/html -t
