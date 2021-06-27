import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

from helpers.pruner import FiltersPruner

import torch
import numpy as np

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq


class HuffmanEncoder:
    def __init__(self, logger):
        self.logger = logger

    # My own self.dump / load logics
    @staticmethod
    def _dump(code_str, filename):
        """
        code_str : string of either '0' and '1' characters
        this function self.dumps to a file
        returns how many bytes are written
        """
        # Make header (1 byte) and add padding to the end
        # Files need to be byte aligned.
        # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
        # This introduces minimum of 8 bits, maximum of 15 bits overhead
        num_of_padding = -len(code_str) % 8
        header = f"{num_of_padding:08b}"
        code_str = header + code_str + '0' * num_of_padding

        # Convert string to integers and to real bytes
        byte_arr = bytearray(int(code_str[i:i + 8], 2) for i in range(0, len(code_str), 8))

        # Dump to a file
        with open(filename, 'wb') as f:
            f.write(byte_arr)
        return len(byte_arr)

    @staticmethod
    def _load(filename):
        """
        This function reads a file and makes a string of '0's and '1's
        """
        with open(filename, 'rb') as f:
            header = f.read(1)
            rest = f.read()  # Bytes
            code_str = ''.join(f'{byte:08b}' for byte in rest)
            offset = ord(header)
            if offset != 0:
                code_str = code_str[:-offset]  # String of '0's and '1's
        return code_str

    # Helper functions for converting between bit string and (float or int)
    @staticmethod
    def _float2bitstr(f):
        four_bytes = struct.pack('>f', f)  # Bytes
        return ''.join(f'{byte:08b}' for byte in four_bytes)  # String of '0's and '1's

    @staticmethod
    def _bitstr2float(bitstr):
        byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
        return struct.unpack('>f', byte_arr)[0]

    @staticmethod
    def _int2bitstr(integer):
        four_bytes = struct.pack('>I', integer)  # Bytes
        return ''.join(f'{byte:08b}' for byte in four_bytes)  # String of '0's and '1's

    @staticmethod
    def _bitstr2int(bitstr):
        byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
        return struct.unpack('>I', byte_arr)[0]

    # Functions for calculating / reconstructing index diff
    @staticmethod
    def _calc_index_diff(indptr):
        return indptr[1:] - indptr[:-1]

    @staticmethod
    def _reconstruct_indptr(diff):
        return np.concatenate([[0], np.cumsum(diff)])

    def _huffman_encode(self, arr, prefix, save_dir='./'):
        """
        Encodes numpy array 'arr' and saves to `save_dir`
        The names of binary files are prefixed with `prefix`
        returns the number of bytes for the tree and the data after the compression
        """
        # Infer dtype
        dtype = str(arr.dtype)

        # Calculate frequency in arr
        freq_map = defaultdict(int)
        convert_map = {'float32': float, 'int32': int}
        for value in np.nditer(arr):
            value = convert_map[dtype](value)
            freq_map[value] += 1

        # Make heap
        heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
        heapify(heap)

        # Merge nodes
        if len(heap) == 1:
            node1 = heappop(heap)
            node2 = Node(0, 0, None, None)
            root = Node(node1.freq, None, node1, node2)
            heappush(heap, root)
        else:
            while len(heap) > 1:
                node1 = heappop(heap)
                node2 = heappop(heap)
                merged = Node(node1.freq + node2.freq, None, node1, node2)
                heappush(heap, merged)

        # Generate code value mapping
        value2code = {}

        def generate_code(node, code):
            if node is None:
                return
            if node.value is not None:
                value2code[node.value] = code
                return
            generate_code(node.left, code + '0')
            generate_code(node.right, code + '1')

        root = heappop(heap)
        generate_code(root, '')

        # Path to save location
        directory = Path(save_dir)

        # Dump data
        data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
        datasize = self._dump(data_encoding, directory/f'{prefix}.bin')

        # Dump codebook (huffman tree)
        codebook_encoding = self._encode_huffman_tree(root, dtype)
        treesize = self._dump(codebook_encoding, directory/f'{prefix}_codebook.bin')

        return treesize, datasize

    def _huffman_decode(self, directory, prefix, dtype):
        """
        Decodes binary files from directory
        """
        directory = Path(directory)

        # Read the codebook
        codebook_encoding = self._load(directory/f'{prefix}_codebook.bin')
        root = self._decode_huffman_tree(codebook_encoding, dtype)

        # Read the data
        data_encoding = self._load(directory/f'{prefix}.bin')

        # Decode
        data = []
        ptr = root
        for bit in data_encoding:
            ptr = ptr.left if bit == '0' else ptr.right
            if ptr.value is not None:  # Leaf node
                data.append(ptr.value)
                ptr = root

        return np.array(data, dtype=dtype)

    # Logics to encode / decode huffman tree
    # Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
    def _encode_huffman_tree(self, root, dtype):
        """
        Encodes a huffman tree to string of '0's and '1's
        """
        converter = {'float32': self._float2bitstr, 'int32': self._int2bitstr}
        code_list = []

        def encode_node(node):
            if node.value is not None:  # Node is leaf node
                code_list.append('1')
                lst = list(converter[dtype](node.value))
                code_list.extend(lst)
            else:
                code_list.append('0')
                encode_node(node.left)
                encode_node(node.right)
        encode_node(root)

        return ''.join(code_list)

    def _decode_huffman_tree(self, code_str, dtype):
        """
        Decodes a string of '0's and '1's and costructs a huffman tree
        """
        converter = {'float32': self._bitstr2float, 'int32': self._bitstr2int}
        idx = 0

        def decode_node():
            nonlocal idx
            info = code_str[idx]
            idx += 1
            if info == '1':  # Leaf node
                value = converter[dtype](code_str[idx:idx+32])
                idx += 32
                return Node(0, value, None, None)
            else:
                left = decode_node()
                right = decode_node()
                return Node(0, None, left, right)

        return decode_node()

    def _huffman_encode_conv(self, param, name, directory, left_dict):
        left_w, left_f, left_c = left_dict[name]

        # Encode
        t0, d0 = self._huffman_encode(left_w, f'{name}_data', directory)
        t1, d1 = self._huffman_encode(left_f, f'{name}_f_indices', directory)
        t2, d2 = self._huffman_encode(left_c, f'{name}_c_indices', directory)

        # Print statistics
        original = param.data.cpu().numpy().nbytes
        compressed = t0 + t1 + t2 + d0 + d1 + d2
        log_text = (
            f"{name:<35} | {original:20} {compressed:20} {original / compressed:>10.2f}x "
            f"{100 * compressed / original:>6.2f}%"
        )
        self.logger.log(log_text, verbose=True)

        return original, compressed

    def _huffman_decode_conv(self, param, name, directory):
        # Decode data
        left_w = self._huffman_decode(directory, f'{name}_data', dtype='float32')
        left_f = self._huffman_decode(directory, f'{name}_f_indices', dtype='int32')
        left_c = self._huffman_decode(directory, f'{name}_c_indices', dtype='int32')

        # Reconstruct weight
        weight = np.zeros(param.shape)
        weight[left_f[:, None], left_c] = left_w.reshape(weight[left_f[:, None], left_c].shape)

        # Return the parameters
        param = torch.from_numpy(weight).to(param.device)
        return param

    def _huffman_encode_fc(self, param, name, directory):
        weight = param.data.cpu().numpy()

        # Encode
        t0, d0 = self._huffman_encode(weight, f'{name}_data', directory)

        # Print statistics
        original = param.data.cpu().numpy().nbytes
        compressed = t0 + d0
        log_text = (
            f"{name:<35} | {original:20} {compressed:20} {original / compressed:>10.2f}x "
            f"{100 * compressed / original:>6.2f}%"
        )
        self.logger.log(log_text, verbose=True)

        return original, compressed

    def _huffman_decode_fc(self, param, name, directory):
        # Decode data
        weight = self._huffman_decode(directory, f'{name}_data', dtype='float32')

        # Reconstruct weight
        weight = weight.reshape(param.shape)

        # Return the parameters
        param = torch.from_numpy(weight).to(param.device)
        return param

    def _direct_dump(self, param, name, directory):
        data = param.data.cpu().numpy()
        data.dump(f'{directory}/{name}')

        # Print statistics
        original = data.nbytes
        compressed = data.nbytes

        log_text = (
            f"{name:<35} | {original:20} {compressed:20} {original / compressed:>10.2f}x "
            f"{100 * compressed / original:>6.2f}%"
        )
        self.logger.log(log_text, verbose=True)

        return original, compressed

    def _direct_load(self, param, name, directory):
        data = np.load(f'{directory}/{name}', allow_pickle=True)
        param = torch.from_numpy(data).to(param.device)
        return param

    # Encode / Decode models
    def huffman_encode_model(self, model, directory='encodings/'):
        def get_title_text():
            return (f"{'Layer':<35} | {'original bytes':>20} {'compressed bytes':>20} {'improvement':>11} "
                    f"{'percent':>7}")
        # Log title
        log_text = (
            f"{get_title_text()}\n" +
            (f"-" * 120)
        )
        self.logger.log(log_text, verbose=True)
        os.makedirs(directory, exist_ok=True)

        # Start Encoding
        # NOTE: It's IMPORTANT to use state_dict() instead of named_parameters() here
        s = {'c': [0, 0], 'f': [0, 0], 'o': [0, 0], 't': [0, 0]}
        s2n = {'c': 'Conv', 'f': 'Fc', 'o': 'Other', 't': 'Total'}
        left_conv_dict = FiltersPruner.get_left_dict(model)
        for name, param in model.state_dict().items():
            if len(param.shape) == 4:
                orig, comp = self._huffman_encode_conv(param, name, directory, left_conv_dict)
                key = 'c'
            elif len(param.shape) == 2:
                orig, comp = self._huffman_encode_fc(param, name, directory)
                key = 'f'
            else:
                orig, comp = self._direct_dump(param, name, directory)
                key = 'o'
            s[key][0] += orig
            s[key][1] += comp
            s['t'][0] += orig
            s['t'][1] += comp

        def get_text_by_key(key):
            nonlocal s, s2n
            name = s2n[key]
            orig = s[key][0]
            comp = s[key][1]
            return f"{name:35} | {orig:>20} {comp:>20} {orig / comp:>10.2f}x {100 * comp / orig:>6.2f}%"

        # Log results of the compression rate
        log_text = (
            f"-" * 120 + "\n" +
            f"{get_text_by_key('t')}\n"
            f"{get_text_by_key('c')}\n"
            f"{get_text_by_key('f')}\n"
            f"{get_text_by_key('o')}\n"
        )
        self.logger.log(log_text, verbose=True)

    def huffman_decode_model(self, model, directory='encodings/'):
        state_dict = dict()
        for name, param in model.state_dict().items():
            if len(param.shape) == 4:
                dec_param = self._huffman_decode_conv(param, name, directory)
            elif len(param.shape) == 2:
                dec_param = self._huffman_decode_fc(param, name, directory)
            else:
                dec_param = self._direct_load(param, name, directory)
            state_dict[name] = dec_param
        model.load_state_dict(state_dict)
