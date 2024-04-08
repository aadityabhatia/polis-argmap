def argdown_heading(label, content=None):
    if content is None:
        return f"\n# {label}\n"
    else:
        return f"\n# {label}\n\n{content}\n"


def argdown_topic(topicName, topicContent, supports=None, tag="AI"):
    emoji = "🤖" if tag == "AI" else "🙂"
    output = f"\n[{topicName}]: {emoji} {topicContent} #{tag}\n"
    if supports is not None:
        return output + argdown_supports(supports)
    return output


def argdown_argument(label, content, supports=None, supportsArgument=None, tag="AI"):
    emoji = "🤖" if tag == "AI" else "🙂"
    output = f"\n<{label}>: {emoji} {content} #{tag}\n"
    if supports is not None:
        return output + argdown_supports(supports)
    if supportsArgument is not None:
        return output + argdown_supports(supportsArgument, argument=True)
    return output


def argdown_comment(label, comment, supports=None, supportsArgument=None, tag="Human"):
    emoji = "🤖" if tag == "AI" else "🙂"
    output = f"\n[{label}]: {emoji} {comment} #{tag}\n"
    if supports is not None:
        return output + argdown_supports(supports)
    if supportsArgument is not None:
        return output + argdown_supports(supportsArgument, argument=True)
    return output


def argdown_supported_by(label):
    return f"  + [{label}]\n"


def argdown_refuted_by(label):
    return f"  - [{label}]\n"


def argdown_supports(label, argument=False):
    if argument:
        return f"  +> <{label}>\n"
    return f"  +> [{label}]\n"


def argdown_refutes(label):
    return f"  -> [{label}]\n"


def argdown_template(output):
    return f"""\
===
sourceHighlighter:
    removeFrontMatter: true
webComponent:
    withoutMaximize: true
    height: 1000px
model:
    removeTagsFromText: true
# map:
#     statementLabelMode: text
#     argumentLabelMode: text
selection:
    excludeDisconnected: false
color:
    tagColors:
        Human: "#99cc99"
        AI: "#9999cc"
===

{output}
"""


def argdown_markdown_template(output):
    return f"""\
```argdown
{argdown_template(output)}
```
"""


def argdown_markdown_map_template(output):
    return f"""\
```argdown-map
{argdown_template(output)}
```
"""
