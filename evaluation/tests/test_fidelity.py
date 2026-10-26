"""System A must stay identical to production *at the baseline commit*.

Sources are read with ``git show <BASELINE_COMMIT>:<path>`` so later edits to
the working tree (Phases 2-6) cannot silently change the frozen baseline.
"""

import ast
import inspect
import subprocess

import pytest

import baseline_a


def _source_at_baseline(path: str) -> str:
    try:
        return subprocess.run(
            ["git", "show", f"{baseline_a.BASELINE_COMMIT}:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"baseline commit {baseline_a.BASELINE_COMMIT[:7]} not available (shallow clone?)")


@pytest.fixture(scope="module")
def tasks_src():
    return _source_at_baseline("worker/tasks.py")


@pytest.fixture(scope="module")
def chat_src():
    return _source_at_baseline("api/routers/chat.py")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _module_constant(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} not found")


def test_extract_text_is_a_verbatim_copy(tasks_src):
    prod = _function(ast.parse(tasks_src), "extract_text")
    frozen = _function(ast.parse(inspect.getsource(baseline_a)), "extract_text")
    assert ast.dump(prod.args) == ast.dump(frozen.args)
    assert [ast.dump(s) for s in prod.body] == [ast.dump(s) for s in frozen.body]


def test_chunking_parameters(tasks_src):
    calls = [n for n in ast.walk(ast.parse(tasks_src))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "RecursiveCharacterTextSplitter"]
    assert len(calls) == 1
    kwargs = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    assert kwargs == {"chunk_size": baseline_a.CHUNK_SIZE, "chunk_overlap": baseline_a.CHUNK_OVERLAP}


@pytest.mark.parametrize("path_fixture", ["tasks_src", "chat_src"])
def test_embedding_model_dim_and_config(path_fixture, request):
    src = request.getfixturevalue(path_fixture)
    tree = ast.parse(src)
    assert ast.literal_eval(_module_constant(tree, "EMBEDDING_DIM")) == baseline_a.EMBEDDING_DIM
    getenv = _module_constant(tree, "EMBEDDING_MODEL")
    assert ast.literal_eval(getenv.args[1]) == baseline_a.EMBEDDING_MODEL
    configs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "EmbedContentConfig"]
    assert configs and all({k.arg for k in c.keywords} == {"output_dimensionality"} for c in configs), \
        "production sets only output_dimensionality (no task_type)"


def test_chat_model_top_k_and_cosine(chat_src):
    tree = ast.parse(chat_src)
    assert ast.literal_eval(_module_constant(tree, "CHAT_MODEL").args[1]) == baseline_a.CHAT_MODEL
    limits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "limit"]
    assert [ast.literal_eval(c.args[0]) for c in limits] == [baseline_a.TOP_K]
    assert "DocumentChunk.embedding.cosine_distance(query_embedding)" in chat_src
    assert "Document.user_id == current_user.id" in chat_src
    assert "<=>" in baseline_a.RETRIEVAL_SQL and "documents.user_id" in baseline_a.RETRIEVAL_SQL


def test_system_prompt_template(chat_src):
    tree = ast.parse(chat_src)
    chat_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "chat")
    assign = next(n for n in ast.walk(chat_fn) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "system_prompt" for t in n.targets))
    assert isinstance(assign.value, ast.JoinedStr)
    template = ""
    for part in assign.value.values:
        if isinstance(part, ast.Constant):
            template += part.value
        else:
            assert isinstance(part, ast.FormattedValue) and part.value.id == "context"
            template += "{context}"
    assert template == baseline_a.SYSTEM_PROMPT_TEMPLATE


def test_context_join_and_generation_call(chat_src):
    assert '"\\n\\n".join([chunk.content for chunk in results])' in chat_src
    assert "generate_content_stream" in chat_src and "system_instruction=system_prompt" in chat_src
    assert baseline_a.build_context(["a", "b"]) == "a\n\nb"
