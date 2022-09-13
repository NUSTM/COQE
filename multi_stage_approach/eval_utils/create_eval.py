from eval_utils.base_eval import ElementEvaluation


def create_first_stage_eval(config, gold_label, gold_sent_label, attn_mask, save_model, fold=0):
    """
    :param config:
    :param gold_label: (elem_label_ids, result_label_ids)
    :param gold_sent_label:
    :param attn_mask:
    :param save_model:
    :param fold:
    :return:
    """
    comparative_identity = True if config.model_type in {"multi_task", "classification"} else False

    res_eval = ElementEvaluation(
        config, fold=fold, target=gold_label, attn_mask=attn_mask, elem_col=config.val.elem_col, save_model=save_model,
        gold_sent_label=gold_sent_label, comparative_identity=comparative_identity
    )

    return res_eval

