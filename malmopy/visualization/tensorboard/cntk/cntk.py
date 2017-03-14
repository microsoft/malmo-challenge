# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import absolute_import

import six
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, tensor_shape_pb2

from ..tensorboard import TensorflowConverter


class CntkConverter(TensorflowConverter):
    def convert(self, network, graph):
        """
        Converts a function from CNTK to the Tensorflow graph format

        Args:
            network: CNTK function that defines the network structure
            graph: destination Tensorflow graph
        """
        # Walk every node of the network iteratively
        stack = [network.model]
        visited = set()

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            try:

                # Function node
                node = node.root_function
                stack.extend(node.inputs)
                try:
                    # TF graph already has the current node
                    graph.get_operation_by_name(node.uid.split('_')[0])
                    continue

                except KeyError:
                    # New network node that has to be converted to TF format
                    # define TF operation attributes based on CNTK network node
                    try:
                        dim_x = tensor_shape_pb2.TensorShapeProto.Dim(size=node.outputs[0].shape[0])
                    except IndexError:
                        dim_x = tensor_shape_pb2.TensorShapeProto.Dim(size=1)
                    try:
                        dim_y = tensor_shape_pb2.TensorShapeProto.Dim(size=node.outputs[0].shape[1])
                    except IndexError:
                        dim_y = tensor_shape_pb2.TensorShapeProto.Dim(size=1)
                    shape = tensor_shape_pb2.TensorShapeProto(dim=(dim_x, dim_y))
                    shape_attr = attr_value_pb2.AttrValue(shape=shape)
                    attrs = {"shape": shape_attr}

                    # Use name scope based on the node's name (e.g. Plus1) to
                    # group the operation and its inputs
                    with graph.name_scope(node.uid) as _:

                        # Create a TF placeholder operation with type, name and shape of the current node
                        op = graph.create_op("Placeholder", inputs=[],
                                             dtypes=[node.outputs[0].dtype], attrs=attrs,
                                             name=node.uid)

                        # Add inputs to the created TF operation
                        for i in six.moves.range(len(node.inputs)):
                            child = node.inputs[i]
                            name = child.uid
                            try:
                                # The input tensor already exists in the graph
                                tf_input = graph.get_tensor_by_name(name + ":0")
                            except KeyError:
                                # A new tensor that needs to be converted from CNTK to TF
                                shape = self.convert_shape(child.shape)
                                dtype = child.dtype
                                # Create a new placeholder tensor with the corresponding attributes
                                tf_input = tf.placeholder(shape=shape, dtype=dtype, name=name)

                            # Update TF operator's inputs
                            op._add_input(tf_input)

                    # Update TF operation's outputs
                    output = node.outputs[0]
                    for o in graph.get_operations():
                        if output.uid in o.name:
                            o._add_input(op.outputs[0])

            except AttributeError:
                # OutputVariable node
                try:
                    if node.is_output:
                        try:
                            # Owner of the node is already added to the TF graph
                            owner_name = node.owner.uid + '/' + node.owner.uid
                            graph.get_operation_by_name(owner_name)
                        except KeyError:
                            # Unknown network node
                            stack.append(node.owner)

                except AttributeError:
                    pass

        # Add missing connections in the graph
        CntkConverter.update_outputs(graph.get_operations())
        graph.finalize()

    @staticmethod
    def convert_shape(shape):
        if len(shape) == 0:
            shape = (1, 1)
        else:
            if len(shape) == 1:
                shape += (1,)
        return shape

    @staticmethod
    def update_outputs(ops):
        """Updates the inputs/outputs of the Tensorflow operations
        by adding missing connections

        Args:
            ops: a list of Tensorflow operations
        """
        for i in six.moves.range(len(ops)):
            for j in six.moves.range(i + 1, len(ops)):
                if ops[i].name.split('/')[1] in ops[j].name.split('/')[1]:
                    ops[i]._add_input(ops[j].outputs[0])
