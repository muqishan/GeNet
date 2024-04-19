from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from lion_pytorch import Lion

# define columns
columns = {0: 'text', 1: 'ner'}
# this is the folder in which train, test and dev files reside
data_folder = 'ner/datasets'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')




# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(model='pubmedbert',
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        tag_format='BIO',
                        use_crf=True,
                        use_rnn=True,
                        reproject_embeddings=True,
                        )

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. run fine-tuning
# 7. 运行微调   resources/taggers/sota-ner-flert

trainer.fine_tune(
    'resources/taggers/sota-ner-flert6',
    learning_rate=5.0e-6,
    mini_batch_size=4,
    max_epochs = 100,
    # optimizer=Lion
    # mini_batch_chunk_size=1,  
)
