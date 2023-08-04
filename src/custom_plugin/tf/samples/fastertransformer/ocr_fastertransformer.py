import tensorflow as tf
import numpy as np
import math
import six
import os
import time
from datetime import datetime
from ocr_transformer_model import fastertransformer_model
from convert_model import load_savedmodel
import json
np.set_printoptions(threshold=1e6, suppress=True)


def load_charset(charsetpath, mode):
    assert mode in ['ctc', 'transformer'], print('invalid charset mode')
    with open(charsetpath) as f:
        lines = f.readlines()
        lines = list(map(lambda x : x.replace('\n', ''), lines))
        chars = [xx for xx in lines if len(xx) > 0]

    if len(chars) == 1:
        chars = list(chars[0])

    if mode == 'ctc':
        chars = chars + ['']
    elif mode == 'transformer':
        if '卍' not in chars:
            chars = ['卍',] + chars
        chars = ['[PAD]', '[EOS]'] + chars
    return chars


class TransformerArgument:
    def __init__(self,
                 batch_size,
                 beam_width,
                 head_num,
                 size_per_head,
                 num_layer,
                 dtype):
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.num_layer = num_layer
        self.dtype = dtype
        self.hidden_dim = self.head_num * self.size_per_head


class DecodingArgument:
  def __init__( self,
                batch_size,
                beam_width,
                head_num,
                size_per_head,
                num_layer,
                vocab_size,
                start_id,
                end_id,
                encoder_hidden_dim,
                dtype):

    self.decoder_args = TransformerArgument(batch_size,
                                            beam_width,
                                            head_num,
                                            size_per_head,
                                            num_layer,
                                            dtype)
    self.vocab_size = vocab_size
    self.start_id = start_id
    self.end_id = end_id
    self.encoder_hidden_dim = encoder_hidden_dim


def load_weights(sess, dtype, variables, variables_dict):
    dic = variables_dict
    fetches = []
    feeds = {}
    var_names = []
    for var in variables:
        var_names.append(var.name)
        if var.name in dic:
            fetches.append(var.initializer)
            feeds[var.initializer.inputs[1]] = dic[var.name].astype(dtype)
        else:
            print(var.name, 'not exsits!')
    sess.run(fetches, feed_dict=feeds)


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def build_transformer(inputs, inputs_shape, chars, tf_datatype, is_resnet_vd):
    encode_num_layer = 3
    encode_head_num = 8
    encode_size_per_head = 64
    encode_hidden_dim = encode_head_num * encode_size_per_head
    decoder_num_layer = 3
    decoder_head_num = 8
    decoder_size_per_head = 64
    vocab_size = len(chars)
    start_of_sentence_id = 0
    end_of_sentence_id = 1
    beam_width = 5
    batch_size = 64

    if is_resnet_vd:
        downsample = 4
    else:
        downsample = 8

    encoder_args = TransformerArgument(batch_size=batch_size,
                                       beam_width=1,
                                       head_num=encode_head_num,
                                       size_per_head=encode_size_per_head,
                                       num_layer=encode_num_layer,
                                       dtype=tf_datatype)

    feats, _ = fastertransformer_model.build_cnn(inputs, encode_hidden_dim, dtype=tf_datatype, is_resnet_vd=is_resnet_vd)

    b = tf.shape(feats)[0]
    w = tf.shape(feats)[1]
    with tf.name_scope("add_pos_encoding"):
        feats += fastertransformer_model.get_position_encoding(w, encode_hidden_dim, dtype=tf_datatype)

    inputs_length = tf.cast(inputs_shape[:,1:2] / downsample, dtype=tf.int32)
    memory_sequence_length = tf.cast(inputs_shape[:,1] / downsample, dtype=tf.int32)
    mask = fastertransformer_model.get_padding(feats, inputs_length, dtype=tf_datatype)
    memory_mask = mask
    memory_mask = tf.expand_dims(memory_mask, axis=1)
    memory_mask = tf.expand_dims(memory_mask, axis=1)
    mask = tf.reshape(mask, [b, w, 1])
    ffn_mask = tf.tile(mask, [1, 1, encode_hidden_dim])
    attention_mask = tf.tile(mask, [1, 1, w])
    attention_mask = tf.transpose(attention_mask, [0, 2, 1])

    # tf encode
    encoder_tf_outputs, _ = fastertransformer_model.tf_encoder(input_tensor=feats,
                                                               encoder_args=encoder_args,
                                                               attention_mask=attention_mask,
                                                               ffn_mask=ffn_mask)
    encoder_tf_outputs = tf.reshape(encoder_tf_outputs, [b, w, encode_hidden_dim])

    # tf decode
    decoding_args = DecodingArgument(batch_size=batch_size,
                                     beam_width=beam_width,
                                     head_num=decoder_head_num,
                                     size_per_head=decoder_size_per_head,
                                     num_layer=decoder_num_layer,
                                     vocab_size=vocab_size,
                                     start_id=start_of_sentence_id,
                                     end_id=end_of_sentence_id,
                                     encoder_hidden_dim=encode_hidden_dim,
                                     dtype=tf_datatype)

    finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, \
        tf_parent_ids, tf_sequence_lengths, cum_log_scores_tf, debug_tf = fastertransformer_model.tf_decoding(encoder_tf_outputs,
                                                                                           memory_sequence_length,
                                                                                           decoding_args)

    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    encoder_var_start_id = 0
    while all_variables[encoder_var_start_id].name.find("conv1d/kerne") == -1:
        encoder_var_start_id += 1
    decoder_var_start_id = 0
    while all_variables[decoder_var_start_id].name.find("embedding_shared_weights") == -1:
        decoder_var_start_id += 1
    encoder_variables = all_variables[encoder_var_start_id:decoder_var_start_id]
    decoder_variables = all_variables[decoder_var_start_id:]

    # op encode
    encoder_op_outputs = fastertransformer_model.op_encoder(inputs=feats,
                                                            encoder_args=encoder_args,
                                                            encoder_vars=encoder_variables,
                                                            attention_mask=attention_mask,
                                                            ffn_mask=ffn_mask)
    encoder_op_outputs = tf.reshape(encoder_op_outputs, [b, w, encode_hidden_dim])

    # op decode
    finalized_op_output_ids, finalized_op_sequence_lengths, op_output_ids, \
        op_parent_ids, op_sequence_lengths, cum_log_scores_op = fastertransformer_model.op_decoding(encoder_op_outputs, memory_sequence_length, decoder_variables, decoding_args)

    # todo 如何判断分数哪个最高
    sentences_id = finalized_tf_output_ids[:, 0, :]
    cum_log_scores_tf = cum_log_scores_tf[:, 0]
    sentences_tf, _ = fastertransformer_model.predicted_ids_with_eos_to_string_v2(sentences_id, chars)

    # todo 如何判断分数哪个最高
    sentences_id = finalized_op_output_ids[:, 0, :]
    cum_log_scores_op = cum_log_scores_op[:, 0]
    sentences_op, _ = fastertransformer_model.predicted_ids_with_eos_to_string_v2(sentences_id, chars)

    return sentences_tf, sentences_op, finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, tf_parent_ids, tf_sequence_lengths, cum_log_scores_tf, debug_tf, finalized_op_output_ids, finalized_op_sequence_lengths, op_output_ids, op_parent_ids, op_sequence_lengths, cum_log_scores_op


def tf_infer_fastertransformer(FP16, src_dir, dst_dir, model_path, chars_path, is_resnet_vd):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    tf_datatype = tf.float32
    np_datatype = np.float32
    if FP16:
        tf_datatype = tf.float16
        np_datatype = np.float16
    chars = load_charset(chars_path, 'transformer')

    # 从savedmodel里面得到权重，重命名权重名，并重置全局默认图
    variables_dict = load_savedmodel(model_path)
    tf.reset_default_graph()

    inputs = tf.placeholder(tf_datatype, shape=([None, 1, 32, None]), name='image')
    inputs_shape = tf.placeholder(tf.int32, shape=([None, 2]), name='image_shape')
    sentences_tf, sentences_op, finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, tf_parent_ids, tf_sequence_lengths, cum_log_scores_tf, debug_tf, finalized_op_output_ids, finalized_op_sequence_lengths, op_output_ids, op_parent_ids, op_sequence_lengths, cum_log_scores_op = build_transformer(inputs, inputs_shape, chars, tf_datatype, is_resnet_vd)

    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    load_weights(sess, np_datatype, all_variables, variables_dict)

    time_tf = 0
    start_cnt = False
    cnt = 0
    names = load_files(src_dir+'/inputs/bin')
    memo = {}
    for name in names:
        bin_name = src_dir + '/inputs/bin/' + name
        shape_name = src_dir + '/inputs/shape/' + name
        s = np.fromfile(shape_name, dtype=np.int32)
        ims = np.fromfile(bin_name,dtype=np.float32).astype(np_datatype).reshape(s)

        imshape_name = src_dir + '/inputs_shape/bin/' + name
        shape_name = src_dir + '/inputs_shape/shape/' + name
        s = np.fromfile(shape_name, dtype=np.int32)
        ims_shape = np.fromfile(imshape_name,dtype=np.int32).reshape(s)

        t = time.time()

        outputs_ = sess.run(sentences_op, feed_dict={inputs:ims, inputs_shape:ims_shape})
        outputs_ = list(map(lambda x : x.decode(), outputs_))
        # outputs_ = sess.run(cum_log_scores_tf, feed_dict={inputs:ims, inputs_shape:ims_shape})
        # print(outputs_)
        # outputs_ = sess.run(cum_log_scores_op, feed_dict={inputs:ims, inputs_shape:ims_shape})
        # print(outputs_)

        time_tf += (time.time() - t) * 1000

        cnt += 1
        if cnt == 10 and not start_cnt:
            start_cnt = True
            time_tf = 0
            cnt = 0

        if start_cnt and cnt > 0 and cnt % 10 == 0:
            print('tf:', time_tf, ' cnt:', cnt, ' tf/per_im:', time_tf/cnt)

        name_name = os.path.join(src_dir, 'name') + '/' + name
        batched_name = np.load(name_name+'.npy')
        for index, image_name in enumerate(batched_name):
            memo[image_name] = {'value' : [outputs_[index]]}

    with open(os.path.join(dst_dir, 'res.txt'), 'w') as f:
        json.dump(memo, f, ensure_ascii=False)

    print('tf:', time_tf, ' cnt:', cnt, ' tf/per_im:', time_tf/cnt)
    sess.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    FP16 = True
    is_resnet_vd = False

    src_dir = '/home/gulixin/workspace/nn_predictor/tensorrt/tests/res/trans_ctc/im_raw_gray_sort_socr'
    if FP16:
        res_dir = 'socr_op_fp16'
    else:
        res_dir = 'socr_op_fp32'
    model_path = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.7/transformer_weights.npz'
    # model_path = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.8-gamma/savedmodel'
    char_path = '/home/gulixin/workspace/datasets/ocr-recognition-data/STD-charset/4pd_charset.txt'
    # char_path = '/home/gulixin/workspace/datasets/ocr-recognition-data/STD-charset/4pd_charset_v2.txt'
    tf_infer_fastertransformer(FP16, src_dir, res_dir, model_path, char_path, is_resnet_vd)


