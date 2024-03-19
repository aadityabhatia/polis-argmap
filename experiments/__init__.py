taskList = [
    'ingest',
    'moderate',
    'topicModel',
    'generate',
    'correlate',
    'score',
    'argmap',
]

def getTask(task):
    if task == 'ingest':
        from .ingest import Ingestion
        return Ingestion
    if task == 'moderate':
        from .moderation import Moderation
        return Moderation
    elif task == 'topicModel':
        from .topicModeling import TopicModeling
        return TopicModeling
    elif task == 'generate':
        from .argumentGeneration import ArgumentGeneration
        return ArgumentGeneration
    elif task == 'correlate':
        from .correlation import Correlation
        return Correlation
    elif task == 'score':
        from .scoring import Scoring
        return Scoring
    elif task == 'argmap':
        from .argMapGeneration import ArgumentMapGeneration
        return ArgumentMapGeneration
    else:
        raise ValueError(f"Unknown Task: {task}")