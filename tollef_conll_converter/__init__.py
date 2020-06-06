from conllu import parse
from conllu.parser import DEFAULT_FIELDS
from collections import defaultdict
import os
import re
import json
import jsonlines
from unidecode import unidecode

def parse_mentions(mentions):
  mention_clusters = []
  for k,v in mentions.items():
    # process the non-clustered mentions:
    non_clusters = [x for x in v if isinstance(x, int)]

    if len(non_clusters) % 2 != 0:
      print("Oops")
      print(k)
      print(v)
      raise EnvironmentError

    # process singleton mentions
    paired_mentions = [x for x in v if isinstance(x, list)]
    # this combining of mentions is safe due to its sequential nature.
    for i in range(0, len(non_clusters), 2):
      paired_mentions.append(non_clusters[i:i+2])

    # we are not interested in mere mention detections (e.g. blue car, no further corefs)
    #if len(paired_mentions) > 1:
    mention_clusters.append(paired_mentions)

  return mention_clusters

def make_jsonline(source, tokens, clusters):
  return {
    "doc_key": source,
    "tokens": tokens,
    "clusters": clusters
  }


# convert all conll-files in a folder specified
class GumConverter(object):
  def __init__(self, folder, output_file, fields=["id", "word", "coref"]):
    self.folder = folder
    self.fields = fields
    self.output = output_file
    self.all_clusters = []
    self.all_tokens = []
    self.singleton_pattern = re.compile(r"\((\w+\d+)\)")

  def compute_clusters(self, data):
    clusters = []
    tokens = []
    for i in range(len(data)):
      words = []

      mentions = defaultdict(list)
      mention_idx = 0

      for s in data[i]:
        word = s["word"]
        words.append(word)
        corefs = s["coref"]
        corefs = corefs.replace("-", "")  # simplify format

        grouped = []
        if corefs and len(corefs) > 1:
          singletons = re.findall(self.singleton_pattern, corefs)
          for st in singletons:
            mentions[st].append([mention_idx, mention_idx])
            complete_st = "({})".format(st)
            # replace it by a separator marker
            corefs = corefs.replace(complete_st, '|')

          corefs = corefs.replace('(', '|').replace(')', '|')
          grouped = [c for c in corefs.split('|') if len(c) > 1]
          for ent in grouped:
              #print(ent)
              mentions[ent].append(mention_idx)
        #print(word, coref, grouped)
        mention_idx += 1

      clusters.extend(parse_mentions(mentions))
      tokens.extend(words)
    return clusters, tokens

  def run(self):
    with jsonlines.open(self.output, mode="w") as writer:
      for filename in os.listdir(self.folder):
        filepath = os.path.join(self.folder, filename)
        print(filepath)
        with open(filepath, "r", encoding="utf8") as f:
          data = f.read()
          conlldata = parse(data, self.fields)
          clusters, tokens = self.compute_clusters(conlldata)

          doc_key = filename.rsplit("\\")[-1]
          one_jsonline = make_jsonline(doc_key, tokens, clusters)
          writer.write(one_jsonline)


class LitBankConverter(object):
  def __init__(self, folder, output_file):
    litbank_fields = list(DEFAULT_FIELDS)
    litbank_fields.insert(0, "sentence_id")
    litbank_fields.insert(0, "litbank_doc")
    litbank_fields.insert(len(litbank_fields), "coref")

    self.fields = litbank_fields

    self.folder = folder
    self.output = output_file
    self.all_clusters = []
    self.all_tokens = []
    self.singleton_pattern = re.compile(r"\((\d+)\)")

  def compute_clusters(self, data):
    clusters = []
    tokens = []

    mention_idx = 0
    mentions = defaultdict(list)
    for sent in data:
      words = []
      
      for field in sent:
        word = field["form"]
        words.append(word)
        if not "coref" in field:
            mention_idx += 1
            continue
        corefs = field["coref"]
        if corefs and len(corefs) > 1:
            singletons = re.findall(self.singleton_pattern, corefs)
            for st in singletons:
                mentions[st].append([mention_idx, mention_idx])
                complete_st = "({})".format(st)
                corefs = corefs.replace(complete_st, '|')
            corefs = corefs.replace('(', '|').replace(')', '|')
            grouped = [c for c in corefs.split('|') if len(c) > 0]
            for ent in grouped:
                mentions[ent].append(mention_idx)
        mention_idx += 1

      tokens.extend(words)

    clusters.extend(parse_mentions(mentions))
    return clusters, tokens

  def run(self):
    with jsonlines.open(self.output, mode="w") as writer:
      for filename in os.listdir(self.folder):
        filepath = os.path.join(self.folder, filename)
        print(filepath)
        with open(filepath, "r", encoding="utf8") as f:
          data = f.read()
          conlldata = parse(data, self.fields)
          clusters, tokens = self.compute_clusters(conlldata)

          doc_key = filename.rsplit("\\")[-1]
          one_jsonline = make_jsonline(doc_key, tokens, clusters)
          writer.write(one_jsonline)

'''
17.04.20
tollef jørgensen

this file converts the format of the PreCo dataset to a CoNLL-compatible jsonline format, as is the output by:
https://github.com/kentonl/e2e-coref/blob/master/minimize.py
'''

class PrecoFormatter():
  def __init__(self, json, coreference_only=False, sent_key="sentences", cluster_key="mention_clusters"):
    self.doc_key = json["id"]
    self.sentences = json[sent_key]
    self.clusters = json[cluster_key]

    # flatten the sentence list
    self.tokens = [t for sublist in self.sentences for t in sublist]

    self.coreference_only = coreference_only

    # some sentences in the dataset are simply whitespace. ignore them.
    self.invalid = None
    self.set_invalid_sents()

    # a 1-to-1 map of sentence index of a token
    self.sentence_map = None  
    self.build_sentence_map()

    self.to_coreflite()

  def set_invalid_sents(self):
    self.invalid = [i for i, sent in enumerate(self.sentences) if len(sent) < 1]

  def build_sentence_map(self):
    idx = 0
    mapping = []
    for sent in self.sentences:
        # add sentence index for the number of tokens in a sentence
        mapping.extend([idx]*len(sent))
        idx += 1
    self.sentence_map = mapping

  def reduce_cluster_mentions_by_index(self, index):
    for c_idx in range(len(self.clusters)):
      for m_idx in range(len(self.clusters[c_idx])):
        try:
          m1, m2 = self.clusters[c_idx][m_idx]
        except Exception:
          print(self.clusters[c_idx][m_idx])
        if m1 > index:
            m1 -= 1
        if m2 > index:
            m2 -= 1
        self.clusters[c_idx][m_idx] = [m1, m2]
  
  def remove_invalid_cluster_tokens(self):
    latex_quotations = ["``"]
    parsed_tokens = []
    for index, token in enumerate(self.tokens):
      if token == " ":
        self.reduce_cluster_mentions_by_index(index)
      else:
        # fix latex quotations.
        if token in latex_quotations:
          token = "''"
        parsed_tokens.append(token)
    self.tokens = parsed_tokens

  def to_coreflite(self):
    if self.coreference_only:
      # remove non-group clusters (i.e. singular mentions)
      self.clusters = [c for c in self.clusters if len(c) > 1]

    def length_of_prev_sentences(sent_index):
      tokencount = len([idx for idx in self.sentence_map if idx < sent_index and idx not in self.invalid])
      # tokencount = len([idx for idx in self.sentence_map if idx < sent_index])
      return tokencount | 0

    coreflite = []
    for cluster in self.clusters:
      coreflite_cluster = []
      for mention in cluster:
        idx, m1, m2 = mention
        prev_token_offset = length_of_prev_sentences(idx)
        x1 = m1 + prev_token_offset
        x2 = m2 + prev_token_offset - 1 # PreCo adds 1 to the end index, remove it.
        coreflite_cluster.append([x1, x2])
      coreflite.append(coreflite_cluster)

    self.clusters = coreflite

    self.remove_invalid_cluster_tokens()

def formatPrecoJsonlines(input_file, output_file):
  with jsonlines.open(output_file, mode='w') as _writer:
    with jsonlines.open(input_file) as _reader:
      for obj in _reader:
        pf = PrecoFormatter(obj)
        jsonobj = make_jsonline(pf.doc_key, pf.tokens, pf.clusters)
        _writer.write(jsonobj)

def formatOntonotesJsonlines(input_file, output_file, genre="all", reverse_genre_logic=False):
  with jsonlines.open(output_file, mode='w') as _writer:
    with jsonlines.open(input_file) as _reader:
      for obj in _reader:
        # flatten the sentence list
        if genre != "all":
          genre_exists = genre in obj["doc_key"]

          if reverse_genre_logic and genre_exists:
            continue
          elif not genre_exists and not reverse_genre_logic:
            continue

        tokens = [t for sublist in obj["sentences"] for t in sublist]
        jsonobj = make_jsonline(obj["doc_key"], tokens, obj["clusters"])
        _writer.write(jsonobj)