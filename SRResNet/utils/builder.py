from .registry import Registry

MODELS = Registry('models')
PIPELINES = Registry('pipelines')
METRICS = Registry('metrics')

def build_pipeline(cfg):
    return PIPELINES.build(cfg)

def build_metrics(cfg):
    return METRICS.build(cfg)

def build_models(cfg):
    return MODELS.build(cfg)
