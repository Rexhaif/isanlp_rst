import os

from allennlp_segmenter import AllenNLPSegmenter
from isanlp.annotation import Sentence

_SEGMENTER = {
    'lstm': AllenNLPSegmenter
}


class ProcessorRST:
    def __init__(self, model_dir_path, segmenter_type='lstm'):

        self._model_dir_path = model_dir_path

        self.segmenter = _SEGMENTER[segmenter_type](self._model_dir_path)

        self._possible_missegmentations = ("\nIMG",
                                           "\nгимнастический коврик;",
                                           "\nгантели или бутылки с песком;",
                                           "\nнебольшой резиновый мяч;",
                                           "\nэластичная лента (эспандер);",
                                           "\nхула-хуп (обруч).",
                                           "\n200?",
                                           "\n300?",
                                           "\nНе требуйте странного.",
                                           "\nИспользуйте мою модель.",
                                           '\n"А чего вы от них требуете?"',
                                           '\n"Решить проблемы с тестерами".',
                                           "\nКак гончая на дичь.",
                                           "\nИ крупная.",
                                           "\nВ прошлом году компания удивила рынок",
                                           "\nЧужой этики особенно.",
                                           "\nНо и своей тоже.",
                                           "\nАэропорт имени,",
                                           "\nА вот и монголы.",
                                           "\nЗолотой Будда.",
                                           "\nДворец Богдо-Хана.",
                                           "\nПлощадь Сухэ-Батора.",
                                           "\nОдноклассники)",
                                           "\nВечерняя площадь.",
                                           "\nТугрики.",
                                           "\nВнутренние монголы.",
                                           "\nВид сверху.",
                                           "\nНациональный парк Тэрэлж. IMG IMG",
                                           '\nГора "Черепаха".',
                                           "\nПуть к медитации.",
                                           "\nЖить надо высоко,",
                                           "\nЧан с кумысом.",
                                           "\nЖилая юрта.",
                                           "\nКумыс.",
                                           "\nТрадиционное занятие монголов",
                                           "\nДвугорбый верблюд мало где",
                                           "\nМонгол Шуудан переводится",
                                           "\nОвощные буузы.",
                                           "\nЗнаменитый чай!",
                                           "\nменя приняли кандидатом",
                                           )

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):

        # 1. Split text and annotations on paragraphs and process separately
        dus = []
        start_id = 0

        for missegmentation in self._possible_missegmentations:
            annot_text = annot_text.replace(missegmentation, ' ' + missegmentation[1:])


        edus = self.segmenter(annot_text, annot_tokens, annot_sentences, annot_lemma,
                                annot_postag, annot_syntax_dep_tree, start_id=start_id)

        return edus