PYTHON ?= python
.PHONY: data train evaluate app test demo

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/build_demo_data.py
train:
	$(PYTHON) scripts/train.py
evaluate:
	$(PYTHON) scripts/evaluate.py
app:
	$(PYTHON) -m streamlit run app/Home.py
test:
	$(PYTHON) -m pytest
demo:
	$(PYTHON) scripts/build_demo_data.py
