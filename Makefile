all:
	\
	python3 -m covid19 --dataAnalytics --trainSVM --trainRF;\

install-dependencies:
	\
	pip3 install --upgrade pip;\
	pip3 install -r requirements.txt;\

data-analytics:
	\
	python3 -m covid19 --dataAnalytics;\

svm:
	\
	python3 -m covid19 --trainSVM;\

svm-data:
	\
	python3 -m covid19 --dataAnalytics --trainSVM;\

rf:
	\
	python3 -m covid19 --trainRF;\

nn:
	\
	python3 -m covid19 --trainNN;\

