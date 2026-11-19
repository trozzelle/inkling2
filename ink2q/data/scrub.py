# calq/data/scrub.py
"""Replace credentials in session text with fixed placeholders before it is rendered or stored."""
from __future__ import annotations

import json
import re

_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)
_KEYS = [
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "<SECRET:jwt>"),
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}"), "<SECRET:key>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "<SECRET:key>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "<SECRET:key>"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), "<SECRET:key>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<SECRET:key>"),
]

# Credential names: (a) anything containing secret/password/passwd, with any _segments after it;
# (a') a name ending in sshpass (SSHPASS=...); (b) a name ending in key/token (optional plural s), or key/token followed only by allowlisted
# suffix segments (_id after key only, so token_id(s) stays plain; _v2, _7, _value, _hash, _salt,
# _secret, _base). key_states, key_cache, token_embeddings, tokens_per_sec, tokenizer do not match.
_NAME = (
    r"(?:[a-z0-9_]*(?:secret|password|passwd)s?(?:_[a-z0-9]+)*|[a-z0-9_]*sshpass"
    r"|[a-z0-9_]*key(?:s?(?:_(?:id|v\d+|\d+|value|hash|salt|secret|base))*)"
    r"|[a-z0-9_]*token(?:s?(?:_(?:v\d+|\d+|value|hash|salt|secret|base))*))(?![a-z0-9_])"
)
_ARG = r"""(?:'[^']*'|"[^"]*"|[^\s'"|]+)"""
_SUDO_PIPE = re.compile(r"(\b(?:echo|printf)\s+(?:-[A-Za-z]+\s+)*)(" + _ARG + r")(\s*\|\s*sudo\s+-S\b)")
_SUDO_HERE = re.compile(r"(\bsudo\s+-S\b[^\n|;&<]{0,300}?<<<[ \t]*)(" + _ARG + r")")
_SAID = re.compile(r"(?i)(\bpassword\s+is\s+)(\S+)")
_SSHPASS = re.compile(r"(\bsshpass\s+(?:-[A-Za-z]\s+)*?-p\s*)(" + _ARG + r")")
_MYSQL_P = re.compile(  # -pSECRET (attached only) on a mysql-family word at line start, after whitespace or a shell separator;
    # a directory prefix counts only when it ends in bin/ (so /var/lib/mysql is a path, not the command)
    r"((?:^|(?<=[\s|;&(`\"']))(?<!name )(?:(?:[.~]?/)?(?:[\w.][\w.-]*/){0,8}s?bin/|\./)?(?:mysql|mysqldump|mysqladmin|mariadb)"
    r"(?:[ \t]+[^\s|;&]{1,200}){0,30}?[ \t]-p)(?=\S)('[^'\n]*'|\"[^\"\n]*\"|[^\s|;&\"'`)]+)", re.M)
_URL_USERINFO = re.compile(  # scheme://user:PASSWORD@host; the password runs to the last @ of the authority
    r"((?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s/:@?#\"'<>,`]*:)([^\s/?#\"'<>,`]+)(@)")
_CURL_USER = re.compile(
    r"(\bcurl\b[^\n|;&]{0,500}?[ \t](?:-u|--user|--proxy-user)(?:[ \t]+|=|(?=[^\s-]))[\"']?[^\s:\"'\\]*:)([^\s\"'\\]+)")
_AUTH = re.compile(
    r"(?i)(\bauthorization\\?[\"']?[ \t]*[:=][ \t]*\\?[\"'`]?)"
    r"(?:((?:bearer|basic|token|negotiate|ntlm)[ \t]+)(?:\$\{[^}\s`]*\}|[^\s\"'\\,`;}]+)"
    r"|(digest[ \t]+)(?:[A-Za-z0-9_*-]+=(?:\\?\"[^\"\n]*?\\?\"|[^\s,\"\\]+)[ \t,]*)+"
    r"|([A-Za-z][\w-]*[ \t]+)(?![A-Za-z_.]+(?:[\s,;}\"'\\`]|$))[^\n\"'\\`]+"  # unknown scheme + credential: rest of the value
    r"|(?![A-Za-z_.]+(?:[\s,;}\"'\\`]|$))[^\s,;}\"'\\`]+)")  # no scheme: one token, unless it is a bare identifier (code)
_SECRET_FLAG = re.compile(
    r"(?i)(?:[a-z0-9]+-)*(?:password|passwd|pass|token|api-?key|secret|client-secret|access-key|secret-key|auth-token|(?<!sort-)(?<!primary-)(?<!partition-)key)")
_FLAG = re.compile(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9-]*)([ \t]+)(\"[^\"\n]*\"|'[^'\n]*'|[^\s\"'-][^\s\"']*)")
_ASSIGN = re.compile(
    r"(?i)(?<![<\w])(" + _NAME + r")"
    r"((?:\\?[\"'])?\]?\s*(?:[=!]==|[=!]=|[=:])\s*)"  # name, optional (escaped) closing quote, subscript bracket, separator
    r"(?:\\\"(?P<edq>(?:[^\"\\\n]|\\\\\\\"|\\[^\"\n])*)\\\""  # \"value\" inside an escaped JSON string
    r"|\"(?P<dq>(?:[^\"\\\n]|\\[^\n])*)\""
    r"|'(?P<sq>(?:[^'\\\n]|\\[^\n])*)'"
    r"|(?P<q>[\"'])(?=[^\s\"',;)\]}])(?P<uq>[^\"'\n]*)"  # unclosed quote: the value runs to the end of the line
    r"|(?P<bare>(?!\\[\"'])[^\s\"',}\[{][^\s\"',}]*))"
)
_BLOCK_HDR = re.compile(r"(?i)^([ \t]*)(?:-[ \t]+)?[\"']?" + _NAME + r"[\"']?[ \t]*:[ \t]*[|>][+-]?\d?[ \t]*$")
_BLOCK_SCALAR = re.compile(r"[|>][+-]?\d?")
_EOL = re.compile(r"[ \t]*(?:\n|\Z)")
_PLAIN = re.compile(r"(?:\d+(?:\.\d+)?|true|false|null|none)", re.I)
_IDENT = re.compile(r"[A-Za-z_]+")
# a name, then dotted segments of letters/underscores (no digits: `abc.def123` is a literal), then an optional call
_EXPR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]+)*(?:\([^\s]*)?[):;\]]*")


def _plain_exempt(name: str, value) -> bool:
    """A value under a credential-like name that is clearly not a secret (max_tokens=4096, key_len: true).
    Names containing pass/secret are never exempt. Non-string scalars (JSON numbers, bools) are always plain."""
    if re.search(r"pass|secret", name, re.I):
        return False
    return not isinstance(value, str) or _PLAIN.fullmatch(value) is not None


def _pw(m: re.Match) -> str:
    arg = m.group(2)
    q = arg[0] if arg[0] in "'\"" else ""
    return f"{m.group(1)}{q}<SECRET:pw>{q}{m.group(3) if m.re is _SUDO_PIPE else ''}"


def _flag(m: re.Match) -> str:
    """`--password hunter2`: a flag whose name is a credential name takes the next word as its secret."""
    flag, gap, arg = m.groups()
    name = flag[2:].replace("-", "_")
    q = arg[0] if arg[0] in "'\"" else ""
    value = arg[1:-1] if q else arg
    if (_SECRET_FLAG.fullmatch(flag[2:]) is None or not value or value.startswith("<SECRET:")
            or not re.search(r"[A-Za-z0-9]", value) or _plain_exempt(name, value)):
        return m.group(0)
    return f"{flag}{gap}{q}<SECRET:value>{q}"


def _in_call(text: str, pos: int) -> bool:
    line = text[text.rfind("\n", 0, pos) + 1 : pos]
    return line.count("(") > line.count(")")


def _assign(m: re.Match) -> str:
    name, sep = m.group(1), m.group(2)
    closing = ""
    if m.group("edq") is not None:
        value, quote, closing = m.group("edq"), '\\"', '\\"'
    elif m.group("dq") is not None:
        value, quote, closing = m.group("dq"), '"', '"'
    elif m.group("sq") is not None:
        value, quote, closing = m.group("sq"), "'", "'"
    elif m.group("uq") is not None:
        value, quote = m.group("uq"), m.group("q")  # stays unclosed
    else:
        value, quote = m.group("bare"), ""
    if not value or value.startswith("<SECRET:"):
        return m.group(0)
    if not quote and sep.startswith("]") and not re.search(r"pass|secret", name, re.I):
        return m.group(0)  # d[key] = value: a subscript with a plain variable key, not an assignment to a credential
    if not quote and _BLOCK_SCALAR.fullmatch(value) and _EOL.match(m.string, m.end()):
        return m.group(0)  # YAML block scalar header; the following lines are scrubbed separately
    if _plain_exempt(name, value):
        return m.group(0)
    if not quote:
        op = sep.rstrip()
        if op.endswith(("==", "!=")):
            if "." in value or "(" in value:
                if _EXPR.fullmatch(value):
                    return m.group(0)  # dotted or call expression (self.key, get_token(0)): code, not a literal
            elif not re.search(r"pass|secret", name, re.I) and _IDENT.fullmatch(value.rstrip("):")):
                return m.group(0)  # token != expected: a plain variable, but never a bare word under pass/secret
        elif not re.search(r"pass|secret", name, re.I) and op.endswith("=") and _in_call(m.string, m.start()):
            if _IDENT.fullmatch(value.rstrip("):")):
                return m.group(0)  # keyword argument bound to a plain variable: sorted(xs, key=len)
    return f"{name}{sep}{quote}<SECRET:value>{closing}"


def _blocks(text: str) -> str:
    out: list[str] = []
    indent: int | None = None
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        eol = line[len(body) :]
        if indent is not None:
            if not body.strip():
                out.append(line)
                continue
            lead = len(body) - len(body.lstrip())
            if lead > indent:
                out.append(f"{body[:lead]}<SECRET:value>{eol}")
                continue
            indent = None
        m = _BLOCK_HDR.match(body)
        if m:
            indent = len(m.group(1))
        out.append(line)
    return "".join(out)


def scrub(text: str) -> str:
    text = _PEM.sub("<SECRET:pem>", text)  # first: key-named fields must not split a PEM block
    text = _blocks(text)
    text = _SUDO_PIPE.sub(_pw, text)
    text = _SUDO_HERE.sub(_pw, text)
    text = _SAID.sub(lambda m: f"{m.group(1)}<SECRET:pw>", text)
    text = _SSHPASS.sub(_pw, text)
    text = _MYSQL_P.sub(lambda m: f"{m.group(1)}<SECRET:pw>", text)
    text = _URL_USERINFO.sub(lambda m: f"{m.group(1)}<SECRET:pw>{m.group(3)}", text)
    text = _CURL_USER.sub(lambda m: f"{m.group(1)}<SECRET:pw>", text)
    text = _AUTH.sub(lambda m: f"{m.group(1)}{m.group(2) or m.group(3) or m.group(4) or ''}<SECRET:value>", text)
    text = _FLAG.sub(_flag, text)
    text = _ASSIGN.sub(_assign, text)  # before _KEYS, so NAME=sk-... becomes <SECRET:value>
    for pat, rep in _KEYS:
        text = pat.sub(rep, text)
    return text


_NAME_FULL = re.compile(_NAME, re.I)


def is_secret_name(key: str) -> bool:
    """True when the whole of `key` is a credential name (api_key, DB_PASSWORD; not tokenizer, key_states)."""
    return _NAME_FULL.fullmatch(key) is not None


def _govern(gov: str | None, key) -> str | None:
    """Governing credential name while descending into `key`: a pass/secret governor always stays; otherwise a
    credential-named key takes over."""
    if gov and re.search(r"pass|secret", gov, re.I):
        return gov
    return key if isinstance(key, str) and is_secret_name(key) else gov


def scrub_obj(obj, _gov: str | None = None):
    """Copy of a decoded JSON value with scrub() applied to every string. Under a credential-named dict key, every
    non-null scalar leaf (str, int, float, bool) in the value, however deeply nested, is replaced whole, except
    plain values `_assign` also keeps (max_tokens: 4096). The outermost credential-named key governs, unless a nested credential name contains pass/secret and the
    governor does not (pass/secret names are never exempt). Other
    leaves are returned unchanged."""
    if isinstance(obj, dict):
        return {k: scrub_obj(v, _govern(_gov, k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v, _gov) for v in obj]
    if _gov and isinstance(obj, (str, int, float)) and not _plain_exempt(_gov, obj):
        return "<SECRET:value>"
    return scrub(obj) if isinstance(obj, str) else obj


def scrub_arguments(value) -> str:
    """Tool-call arguments as a JSON string with every decoded string value scrubbed (never the serialized JSON).
    Decodes up to two JSON levels, as the encoder does; a non-JSON string falls back to plain-text scrub()."""
    if isinstance(value, str):
        text, decoded = value, False
        for _ in range(2):
            if not isinstance(value, str):
                break
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                break
            decoded = True
        if not decoded:
            return scrub(text)
    return json.dumps(scrub_obj(value), ensure_ascii=False)
