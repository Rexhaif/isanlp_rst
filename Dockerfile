FROM inemo/isanlp_base_cuda

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y liblzma-dev

ENV PYENV_ROOT /opt/.pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /opt/.pyenv/shims:/opt/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
RUN pyenv install 3.6.4
RUN pyenv global 3.6.4

RUN pip install -U pip
RUN python -m pip install -U cython

RUN pip install dostoevsky
RUN dostoevsky download fasttext-social-network-model

RUN pip install setuptools==41.0.1 scipy scikit-learn==0.22.1 gensim==3.6.0 smart-open==1.7.0 tensorflow==2.1.0 keras h5py tensorflow-hub pandas nltk imbalanced-learn catboost

RUN pip install allennlp==0.9.0
RUN pip install -U git+https://github.com/IINemo/isanlp.git@discourse

RUN python -c "import nltk; nltk.download('stopwords')"

COPY src/isanlp_rst /src/isanlp_rst
COPY pipeline_object.py /src/isanlp_rst/pipeline_object.py
COPY models /models

### Uncomment this section if embedders are not in the current directory
## ELMo
#RUN curl -O http://vectors.nlpl.eu/repository/20/195.zip && unzip 195.zip && rm 195.zip
#RUN mkdir rsv_elmo && mv model.hdf5 rsv_elmo/model.hdf5 && mv options.json rsv_elmo/options.json
## RuBERT
#RUN curl -O http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz && tar -xzvf rubert_cased_L-12_H-768_A-12_pt.tar.gz && rm rubert_cased_L-12_H-768_A-12_pt.tar.gz
## fastText embeddings
#RUN curl -O http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec

# COPY rsv_elmo /rsv_elmo
# COPY rubert_cased_L-12_H-768_A-12_pt /rubert_cased_L-12_H-768_A-12_pt
RUN curl -O https://rexhaif.keybase.pub/model.tar.gz && mkdir -p /models/segmenter_neural/ && mv model.tar.gz /models/segmenter_neural/model.tar.gz

## Check RuBERT
RUN python -c "from allennlp.predictors import Predictor; predictor = Predictor.from_path('models/segmenter_neural/model.tar.gz')"

ENV PYTHONPATH=/src/isanlp_rst/
CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]
