all:
	python3 -m covid19

install-dependencies:
	\
	pip3 install --upgrade pip;\
	pip3 install -r requirements.txt;\

svm:
	python3 -m covid19 --trainSVM

