"""Tiny helpers for building shader / geometry node graphs from Python.

    g = NodeBuilder(material.node_tree)
    d = g.math("MULTIPLY", g.value(3.912), g.math("DIVIDE", 1.0, vis))
    g.link(d, volume.inputs["Density"])

Every helper accepts either a float (turned into an input default) or a node
socket, so graphs read like expressions.
"""
from __future__ import annotations

from typing import Union

import bpy

Sock = Union[float, int, bpy.types.NodeSocket]


class NodeBuilder:
    def __init__(self, tree: bpy.types.NodeTree, clear: bool = False):
        self.tree = tree
        self._x = 0
        if clear:
            for n in list(tree.nodes):
                tree.nodes.remove(n)

    # ---- raw -------------------------------------------------------------
    def node(self, idname: str, label: str = None, **props):
        n = self.tree.nodes.new(idname)
        self._x -= 40
        n.location = (self._x, (len(self.tree.nodes) % 25) * -60)
        if label:
            n.label = label
            n.name = label
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        if isinstance(a, bpy.types.Node):
            a = a.outputs[0]
        self.tree.links.new(a, b)

    def set(self, sock: bpy.types.NodeSocket, v: Sock):
        """Connect a socket to v (socket) or set its default value (number / tuple)."""
        if isinstance(v, bpy.types.Node):
            v = v.outputs[0]
        if isinstance(v, bpy.types.NodeSocket):
            self.tree.links.new(v, sock)
        else:
            if hasattr(sock, "default_value"):
                dv = sock.default_value
                if hasattr(dv, "__len__") and not hasattr(v, "__len__"):
                    v = [v] * len(dv)
                    if len(dv) == 4:
                        v[3] = 1.0
                sock.default_value = v

    # ---- math ------------------------------------------------------------
    def math(self, op: str, a: Sock, b: Sock = 0.0, c: Sock = 0.0, clamp=False, label=None):
        n = self.node("ShaderNodeMath", label=label, operation=op, use_clamp=clamp)
        self.set(n.inputs[0], a)
        if len(n.inputs) > 1:
            self.set(n.inputs[1], b)
        if len(n.inputs) > 2:
            self.set(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)
    def clamp01(self, a): return self.math("ADD", a, 0.0, clamp=True)

    def value(self, v: float, label: str = None):
        n = self.node("ShaderNodeValue", label=label)
        n.outputs[0].default_value = v
        return n.outputs[0]

    def vmath(self, op: str, a: Sock, b: Sock = None, scale: float = None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if scale is not None and "Scale" in n.inputs:
            n.inputs["Scale"].default_value = scale
        out = n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]
        return out

    def separate(self, v: Sock):
        n = self.node("ShaderNodeSeparateXYZ")
        self.set(n.inputs[0], v)
        return n.outputs["X"], n.outputs["Y"], n.outputs["Z"]

    def combine(self, x: Sock, y: Sock, z: Sock):
        n = self.node("ShaderNodeCombineXYZ")
        self.set(n.inputs[0], x)
        self.set(n.inputs[1], y)
        self.set(n.inputs[2], z)
        return n.outputs[0]

    def smoothstep(self, e0: float, e1: float, x: Sock):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self.set(n.inputs["Value"], x)
        n.inputs["From Min"].default_value = e0
        n.inputs["From Max"].default_value = e1
        return n.outputs["Result"]

    def maprange(self, x: Sock, a0, a1, b0, b1, clamp=True):
        n = self.node("ShaderNodeMapRange", clamp=clamp)
        self.set(n.inputs["Value"], x)
        n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = a0, a1
        n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = b0, b1
        return n.outputs["Result"]

    def mix_color(self, fac: Sock, a: Sock, b: Sock, blend="MIX"):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.set(n.inputs["Factor"], fac)
        self.set(n.inputs["A"], a)
        self.set(n.inputs["B"], b)
        return n.outputs["Result"]

    def rgb(self, color):
        n = self.node("ShaderNodeRGB")
        n.outputs[0].default_value = (*color[:3], 1.0)
        return n.outputs[0]

    def attr(self, name: str, kind: str = "VIEW_LAYER", out: str = "Fac"):
        n = self.node("ShaderNodeAttribute", attribute_type=kind, attribute_name=name)
        return n.outputs[out]
