import numpy as np
import pandas as pd

from isanlp.annotation_rst import DiscourseUnit


class GreedyRSTParser:
    def __init__(self, tree_predictor, confidence_threshold=0.05):
        """
        :param RSTTreePredictor tree_predictor:
        :param float confidence_threshold: minimum relation probability to append the pair into the tree
        """
        self.tree_predictor = tree_predictor
        self.confidence_threshold = confidence_threshold

    def __call__(self, edus, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree, genre=None):
        """
        :param list edus: DiscourseUnit
        :param str annot_text: original text
        :param list annot_tokens: isanlp.annotation.Token
        :param list annot_sentences: isanlp.annotation.Sentence
        :param list annot_postag: lists of str for each sentence
        :param annot_lemma: lists of str for each sentence
        :param annot_syntax_dep_tree: list of isanlp.annotation.WordSynt for each sentence
        :return: list of DiscourseUnit containing each extracted tree
        """

        def to_merge(_scores):
            return np.argmax(np.array(_scores))

        self.tree_predictor.genre = genre

        nodes = edus

        max_id = edus[-1].id

        # initialize scores
        features = self.tree_predictor.initialize_features(nodes, 
                                                           annot_text, annot_tokens,
                                                           annot_sentences,
                                                           annot_lemma, annot_morph, annot_postag,
                                                           annot_syntax_dep_tree)

        scores = self.tree_predictor.predict_pair_proba(features)

        while len(nodes) > 2 and any([score > self.confidence_threshold for score in scores]):
            # select two nodes to merge
            j = to_merge(scores)  # position of the pair in list

            # make the new node by merging node[j] + node[j+1]
            relation = self.tree_predictor.predict_label(features.iloc[j]),
            relation, nuclearity = relation[0].split('_')
            temp = DiscourseUnit(
                id=max_id + 1,
                left=nodes[j],
                right=nodes[j + 1],
                relation=relation,
                nuclearity=nuclearity,
                proba=scores[j],
                text=annot_text[nodes[j].start:nodes[j + 1].end].strip()
            )
            
            print('NEW UNIT')
            print(temp)

            max_id += 1

            # modify the node list
            nodes = nodes[:j] + [temp] + nodes[j + 2:]

            # modify the scores list
            if j == 0:
                print('LEFT: [nodes[j], nodes[j + 1]]:', [nodes[j].text, nodes[j + 1].text])
                _features = self.tree_predictor.extract_features(nodes[j], nodes[j + 1],
                                                                 annot_text, annot_tokens,
                                                                 annot_sentences,
                                                                 annot_lemma, annot_morph, annot_postag,
                                                                 annot_syntax_dep_tree)
                print('LEFT: _features:', _features)
                _scores = self.tree_predictor.predict_pair_proba(_features)
                scores = _scores + scores[j + 2:]
                features = pd.concat([_features, features.iloc[j + 2:]])

            elif j + 1 < len(nodes):
                print('MIDDLE: [nodes[j - 1], nodes[j], nodes[j + 1]]:', [nodes[j - 1], nodes[j], nodes[j + 1]])
                _features = self.tree_predictor.initialize_features([nodes[j - 1], nodes[j], nodes[j + 1]],
                                                                    annot_text, annot_tokens,
                                                                    annot_sentences,
                                                                    annot_lemma, annot_morph, annot_postag,
                                                                    annot_syntax_dep_tree)
                print('MIDDLE: _features:', _features)
                _scores = self.tree_predictor.predict_pair_proba(_features)
                features = pd.concat([features.iloc[:j - 1], _features, features.iloc[j + 2:]])
                scores = scores[:j - 1] + _scores + scores[j + 2:]

            else:
                print('LEFT: [nodes[j - 1], nodes[j]]:', [nodes[j - 1], nodes[j]])
                _features = self.tree_predictor.extract_features(nodes[j - 1], nodes[j],
                                                                 annot_text, annot_tokens,
                                                                 annot_sentences,
                                                                 annot_lemma, annot_morph, annot_postag,
                                                                 annot_syntax_dep_tree)
                print('RIGHT: _features:', _features)
                _scores = self.tree_predictor.predict_pair_proba(_features)
                scores = scores[:j - 1] + _scores
                features = pd.concat([features.iloc[:j - 1], _features])

        relation = self.tree_predictor.predict_label(features.iloc[0]),
        relation, nuclearity = relation[0].split('_')
        if len(scores) == 1 and scores[0] > self.confidence_threshold:
            root = DiscourseUnit(
                id=max_id + 1,
                left=nodes[0],
                right=nodes[1],
                relation=relation,
                nuclearity=nuclearity,
                proba=scores[0],
                text=annot_text[nodes[0].start:nodes[1].end].strip()
            )
            nodes = [root]

        return nodes
