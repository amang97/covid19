all:
	\
	python3 -m covid19 --trainSVM --trainRF;\

install-dependencies:
	\
	pip3 install --upgrade pip;\
	pip3 install -r requirements.txt;\

svm:
	\
	python3 -m covid19 --trainSVM;\

rf:
	\
	python3 -m covid19 --trainRF;\

nn:
	\
	python3 -m covid19 --trainNN;\

