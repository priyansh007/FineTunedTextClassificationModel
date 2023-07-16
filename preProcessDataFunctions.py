import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from imblearn.over_sampling import RandomOverSampler #Over sampling
from imblearn.under_sampling import RandomUnderSampler #Under sampling
import re
import contractions
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def remove_stopwords(sentences):
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []

    for sentence in sentences:
      words = word_tokenize(sentence)
      temp = [word for word in words if word not in stop_words]
      filtered_sent = ' '.join(temp)
      filtered_sentences.append(filtered_sent)

    return filtered_sentences

def remove_punctuation(sentences):
    punctuation = string.punctuation
    filtered_sentences = []

    for sentence in sentences:
      temp = [char for char in sentence if char not in punctuation]
      new_sent = ''.join(temp)
      filtered_sentences.append(new_sent)
    return filtered_sentences

def stem_sentences(sentences):
    stemmer = PorterStemmer()
    stemmed_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        temp = [stemmer.stem(word) for word in words]
        stemmed_sent = ' '.join(temp)
        stemmed_sentences.append(stemmed_sent)
    return stemmed_sentences

def lem_sentences(sentences):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        temp = [lemmatizer.lemmatize(word) for word in words]
        stemmed_sent = ' '.join(temp)
        lemmatized_sentences.append(stemmed_sent)

    return lemmatized_sentences

def preprocess_sentence(sentences):
    # Remove punctuation
    sentences = remove_punctuation(sentences)
    sentences = remove_stopwords(sentences)
    sentences = stem_sentences(sentences)
    sentences = lem_sentences(sentences)
    return sentences

def clean_content(text):

  # remove twitter handles
  clean_text = re.sub(r'@\w+\s?', "", str(text))

  # convert to lowercase
  clean_text = clean_text.lower()

  # remove links http:// or https://
  clean_text = re.sub(r'https?:\/\/\S+', '', clean_text)

  # remove links beginning with www. and ending with .com
  clean_text = re.sub(r'www\.[a-z]?\.?(com)+|[a-z]+\.(com)', '', clean_text)

  # remove html reference characters
  clean_text = re.sub(r'&[a-z]+;', '', clean_text)

  # remove non-letter characters besides spaces "/", ";" "[", "]" "=", "#" Regex syntax
  clean_text = re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', clean_text)

  clean_text = contractions.fix(clean_text)
  sentences = sent_tokenize(clean_text)
  filtered_sentences = preprocess_sentence(sentences)
  clean_text = '.'.join(filtered_sentences)
  return clean_text

def balanceDataframe(X_values, y_values):
    class_counts = y_values.value_counts()
    categories = list(class_counts.index)
    average_count = int(class_counts.mean())
    oversampler = RandomOverSampler(random_state=42, 
                                    sampling_strategy= { element:average_count for element in [categories[index] for index, i in enumerate(class_counts) if i < average_count]})
    x_over_sample, y_over_sample = oversampler.fit_resample(X_values, y_values)
    undersampler = RandomUnderSampler(random_state=42, 
                                      sampling_strategy={ element:average_count for element in [categories[index] for index, i in enumerate(class_counts) if i >= average_count]})
    X_balanced, y_balanced  = undersampler.fit_resample(x_over_sample, y_over_sample)
    return X_balanced, y_balanced