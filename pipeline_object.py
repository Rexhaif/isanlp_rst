from processor_rst import ProcessorRST
from isanlp import PipelineCommon


def create_pipeline(delay_init):
    pipeline_default = PipelineCommon([(ProcessorRST(model_dir_path='/models',
                                                     segmenter_type='lstm'),
                                        ['text', 'tokens', 'sentences', 'lemma', 'morph', 'postag', 'syntax_dep_tree'],
                                        {0: 'rst'})
                                       ],
                                      name='default')

    return pipeline_default
