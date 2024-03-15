import guidance
from guidance import gen, select


@guidance(stateless=True)
def generate_line(lm, name: str, temperature=0, max_tokens=50, list_append=False):
    return lm + gen(name=name, max_tokens=max_tokens, temperature=temperature, list_append=list_append, stop=['\n'])


@guidance(stateless=True)
def generate_phrase(lm, name: str, temperature=0, max_tokens=50, list_append=False):
    return lm + gen(name=name, max_tokens=max_tokens, temperature=temperature, list_append=list_append, stop=['\n', '.'])


@guidance(stateless=True)
def generate_number(lm, name: str, min: int, max: int, list_append=False):
    return lm + select(list(range(min, max+1)), name=name, list_append=list_append)
