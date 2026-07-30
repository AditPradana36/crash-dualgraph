"""
HeteroConv+GATv2Conv encoders and fusion heads for scenarios A-G.

Functions/classes to implement:
- class SVGEncoder(nn.Module): ...
- class TVGEncoder(nn.Module): ...
- class FusionHead(nn.Module): ...   # parametrized by scenario (concat/late/cross-attn/merged)
- build_model(scenario: str, config: dict) -> nn.Module
"""
