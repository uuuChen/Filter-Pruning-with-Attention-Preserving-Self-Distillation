import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

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

    def _huffman_encode_conv(self, param, name, directory):
        #################################
        # TODO:
        #   You can refer to the code of the function "_huffman_encode_fc" below, but note that "csr_matrix" can only
        #   be used on 2-dimensional data
        #   --------------------------------------------------------
        #   HINT:
        #   Suppose the shape of the weights of a certain convolution layer is (Kn, Ch, W, H)
        #   ---
        #   1. Call function "csr_matrix" for all (Kn * Ch) two-dimensional matrices (W, H), and get "data",
        #   "length of data", "indices", and "indptr" of all (Kn * Ch) csr_matrix.
        #   2. Concatenate these 4 parts of all (Kn * Ch) csr_matrices individually into 4 one-dimensional
        #   lists, so there will be 4 lists.
        #   3. Do huffman coding on these 4 lists individually.
        #################################

        # Note that we do not huffman encode "conv" yet. The following four lines of code need to be modified
        conv = param.data.cpu().numpy()
        conv.dump(f'{directory}/{name}')

        # Print statistics
        original = conv.nbytes
        compressed = original
        log_text = (
            f"{name:<15} | "
            f"{original:20} {compressed:20} {original / compressed:>10.2f}x "
            f"{100 * compressed / original:>6.2f}% (NEED TO BE IMPLEMENTED)"
        )
        self.logger.log(log_text, verbose=True)

        return original, compressed

    def _huffman_encode_fc(self, param, name, directory):
        weight = param.data.cpu().numpy()
        shape = weight.shape

        form = 'csr' if shape[0] < shape[1] else 'csc'
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

        # Encode
        t0, d0 = self._huffman_encode(mat.data, name + f'_{form}_data', directory)
        t1, d1 = self._huffman_encode(mat.indices, name + f'_{form}_indices', directory)
        t2, d2 = self._huffman_encode(self._calc_index_diff(mat.indptr), name + f'_{form}_indptr', directory)

        # Print statistics
        original = param.data.cpu().numpy().nbytes
        compressed = t0 + t1 + t2 + d0 + d1 + d2
        log_text = (
            f"{name:<15} | {original:20} {compressed:20} {original / compressed:>10.2f}x "
            f"{100 * compressed / original:>6.2f}%"
        )
        self.logger.log(log_text, verbose=True)

        return original, compressed

    def _dump_bias(self, param, name, directory):
        # Note that we do not huffman encode bias
        bias = param.data.cpu().numpy()
        bias.dump(f'{directory}/{name}')

        # Print statistics
        original = bias.nbytes
        compressed = bias.nbytes

        log_text = (
            f"{name:<15} | "
            f"{original:20} {compressed:20} {original / compressed:>10.2f}x "
            f"{100 * compressed / original:>6.2f}%"
        )
        self.logger.log(log_text, verbose=True)

        return original, compressed

    def _huffman_decode_conv(self, param, name, directory):
        #################################
        # TODO:
        #   Decode according to the code of "conv" section you write in the function "huffman encode model"
        #   above, and refer to encode and decode code of "fc"
        #################################

        # Note that we do not huffman decode "conv" yet. The following three lines of code need to be modified
        conv = np.load(directory + '/' + name, allow_pickle=True)
        param.data = torch.from_numpy(conv).to(param.device)

    def _huffman_decode_fc(self, param, name, directory):
        weight = param.data.cpu().numpy()
        shape = weight.shape

        form = 'csr' if shape[0] < shape[1] else 'csc'
        matrix = csr_matrix if shape[0] < shape[1] else csc_matrix

        # Decode data
        data = self._huffman_decode(directory, name + f'_{form}_data', dtype='float32')
        indices = self._huffman_decode(directory, name + f'_{form}_indices', dtype='int32')
        indptr = self._reconstruct_indptr(self._huffman_decode(directory, name + f'_{form}_indptr', dtype='int32'))

        # Construct matrix
        mat = matrix((data, indices, indptr), shape)

        # Insert to model
        param.data = torch.from_numpy(mat.toarray()).to(param.device)

    def _load_bias(self, param, name, directory):
        bias = np.load(directory + '/' + name, allow_pickle=True)
        param.data = torch.from_numpy(bias).to(param.device)

    # Encode / Decode models
    def huffman_encode_model(self, model, directory='encodings/'):
        os.makedirs(directory, exist_ok=True)
        original_sum = 0
        compressed_sum = 0
        log_text = (f"{'Layer':<15} | {'original bytes':>20} {'compressed bytes':>20} {'improvement':>11} "
                    f"{'percent':>7}\n"
                    f"-" * 70)
        self.logger.log(log_text, verbose=True)
        for name, param in model.named_parameters():
            if 'weight' in name:  # Weights
                if 'conv' in name:
                    original, compressed = self._huffman_encode_conv(param, name, directory)
                elif 'fc' in name:
                    original, compressed = self._huffman_encode_fc(param, name, directory)
                else:
                    raise NameError
            else:  # Bias
                original, compressed = self._dump_bias(param, name, directory)
            original_sum += original
            compressed_sum += compressed
        log_text = (f"-" * 70 + "\n" 
                    f"{'total':15} | {original_sum:>20} {compressed_sum:>20} {original_sum / compressed_sum:>10.2f}x "
                    f"{100 * compressed_sum / original_sum:>6.2f}%")
        self.logger.log(log_text, verbose=True)

    def huffman_decode_model(self, model, directory='encodings/'):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    self._huffman_decode_conv(param, name, directory)
                elif 'fc' in name:
                    self._huffman_decode_fc(param, name, directory)
                else:
                    raise NameError
            else:
                self._load_bias(param, name, directory)
