"""
Testes para src/chat.py — Story 4: CLI de Chat

Spec completa: docs/specs/2026-04-03-tdd-spec-langchain-rag.md
"""

from unittest.mock import MagicMock, call

import pytest

import chat


def _make_chain(responses=None, empty_collection=False):
    chain = MagicMock()

    if responses is None:
        chain.side_effect = ["Resposta baseada no PDF."]
    else:
        chain.side_effect = responses

    chain.is_empty = MagicMock(return_value=empty_collection)
    return chain


def test_chat_startup_failure_prints_error_and_exits(mocker, capsys):
    input_mock = mocker.patch(
        "builtins.input",
        side_effect=AssertionError("input não deveria ser chamado"),
    )
    mocker.patch("chat.search_prompt", side_effect=RuntimeError("falha ao iniciar"))

    chat.main()

    captured = capsys.readouterr()
    assert "Não foi possível iniciar o chat:" in captured.out
    assert "falha ao iniciar" in captured.out
    assert input_mock.call_count == 0


def test_chat_prints_banner_once_before_prompting(mocker, capsys):
    chain = _make_chain()
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch("builtins.input", side_effect=["sair"])

    chat.main()

    captured = capsys.readouterr()
    assert captured.out.count("Faça sua pergunta:") == 1
    assert input_mock.call_args_list == [call("PERGUNTA: ")]


def test_chat_calls_search_prompt_once_before_entering_loop(mocker):
    chain = _make_chain()
    search_prompt_mock = mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch("builtins.input", side_effect=["sair"])

    chat.main()

    search_prompt_mock.assert_called_once_with()
    chain.is_empty.assert_called_once_with()


def test_chat_uses_pergunta_prompt_and_resposta_prefix_for_valid_question(mocker, capsys):
    chain = _make_chain(responses=["Resposta de teste"])
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch("builtins.input", side_effect=["O que é LangChain?", "sair"])

    chat.main()

    captured = capsys.readouterr()
    chain.assert_called_once_with("O que é LangChain?")
    assert input_mock.call_args_list[0] == call("PERGUNTA: ")
    assert "RESPOSTA: Resposta de teste" in captured.out


def test_chat_chain_called_exactly_once_per_valid_question(mocker):
    chain = _make_chain(responses=["Resposta única"])
    mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch("builtins.input", side_effect=["pergunta válida", "sair"])

    chat.main()

    chain.assert_called_once_with("pergunta válida")


@pytest.mark.parametrize("command", ["sair", "exit", "quit", "SAIR", "Exit", "QUIT"])
def test_chat_exits_cleanly_for_exit_commands_without_calling_chain(mocker, command):
    chain = _make_chain()
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch("builtins.input", side_effect=[command])

    chat.main()

    chain.assert_not_called()
    assert input_mock.call_args_list == [call("PERGUNTA: ")]


def test_chat_ctrl_c_exits_without_traceback(mocker, capsys):
    chain = _make_chain()
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch("builtins.input", side_effect=KeyboardInterrupt)

    chat.main()

    captured = capsys.readouterr()
    assert input_mock.call_args_list == [call("PERGUNTA: ")]
    assert "Traceback" not in captured.err
    chain.assert_not_called()


@pytest.mark.parametrize("raw_input", ["", "   ", "\n\n"])
def test_chat_ignores_empty_and_whitespace_inputs_without_printing_resposta(mocker, capsys, raw_input):
    chain = _make_chain(responses=["Resposta válida"])
    mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch("builtins.input", side_effect=[raw_input, "pergunta útil", "sair"])

    chat.main()

    captured = capsys.readouterr()
    chain.assert_called_once_with("pergunta útil")
    assert captured.out.count("RESPOSTA:") == 1


def test_chat_truncates_inputs_over_500_chars_and_warns(mocker, capsys):
    chain = _make_chain(responses=["Resposta curta"])
    mocker.patch("chat.search_prompt", return_value=chain)
    long_question = "a" * 501
    mocker.patch("builtins.input", side_effect=[long_question, "sair"])

    chat.main()

    captured = capsys.readouterr()
    chain.assert_called_once_with(long_question[:500])
    assert "500" in captured.out
    assert "trunc" in captured.out.lower()


def test_chat_does_not_truncate_or_warn_at_exactly_500_chars(mocker, capsys):
    chain = _make_chain(responses=["Resposta exata"])
    mocker.patch("chat.search_prompt", return_value=chain)
    question = "b" * 500
    mocker.patch("builtins.input", side_effect=[question, "sair"])

    chat.main()

    captured = capsys.readouterr()
    chain.assert_called_once_with(question)
    assert "trunc" not in captured.out.lower()


def test_chat_does_not_truncate_or_warn_below_500_chars(mocker, capsys):
    chain = _make_chain(responses=["Resposta 499"])
    mocker.patch("chat.search_prompt", return_value=chain)
    question = "c" * 499
    mocker.patch("builtins.input", side_effect=[question, "sair"])

    chat.main()

    captured = capsys.readouterr()
    chain.assert_called_once_with(question)
    assert "trunc" not in captured.out.lower()


def test_chat_warns_and_exits_when_collection_is_empty_before_loop(mocker, capsys):
    chain = _make_chain(empty_collection=True)
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch(
        "builtins.input",
        side_effect=AssertionError("input não deveria ser chamado"),
    )

    chat.main()

    captured = capsys.readouterr()
    assert captured.out.strip()
    assert input_mock.call_count == 0
    chain.is_empty.assert_called_once_with()


def test_chat_collection_probe_failure_prints_error_and_exits(mocker, capsys):
    chain = _make_chain()
    chain.is_empty.side_effect = RuntimeError("falha no probe")
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch(
        "builtins.input",
        side_effect=AssertionError("input não deveria ser chamado"),
    )

    chat.main()

    captured = capsys.readouterr()
    assert "Não foi possível iniciar o chat:" in captured.out
    assert "falha no probe" in captured.out
    assert input_mock.call_count == 0


def test_chat_ctrl_c_during_chain_execution_exits_without_traceback(mocker, capsys):
    chain = _make_chain(responses=[KeyboardInterrupt()])
    mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch("builtins.input", side_effect=["pergunta interrompida"])

    chat.main()

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "RESPOSTA:" not in captured.out


def test_chat_eof_exits_without_traceback(mocker, capsys):
    chain = _make_chain()
    mocker.patch("chat.search_prompt", return_value=chain)
    input_mock = mocker.patch("builtins.input", side_effect=EOFError)

    chat.main()

    captured = capsys.readouterr()
    assert input_mock.call_args_list == [call("PERGUNTA: ")]
    assert "Traceback" not in captured.err
    chain.assert_not_called()


def test_chat_logs_runtime_error_and_continues_session(mocker, capsys):
    chain = _make_chain(responses=[RuntimeError("connection lost"), "Resposta após recuperação"])
    mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch("builtins.input", side_effect=["primeira pergunta", "segunda pergunta", "sair"])

    chat.main()

    captured = capsys.readouterr()
    assert "connection lost" in captured.out
    assert "RESPOSTA: Resposta após recuperação" in captured.out
    assert chain.call_count == 2


def test_chat_does_not_print_resposta_for_failed_question(mocker, capsys):
    chain = _make_chain(responses=[RuntimeError("connection lost"), "Resposta após recuperação"])
    mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch("builtins.input", side_effect=["primeira pergunta", "segunda pergunta", "sair"])

    chat.main()

    captured = capsys.readouterr().out.splitlines()
    assert "RESPOSTA: None" not in "\n".join(captured)
    assert captured.count("RESPOSTA: Resposta após recuperação") == 1


def test_chat_multiple_questions_answered_in_sequence(mocker, capsys):
    chain = _make_chain(responses=["Resposta 1", "Resposta 2", "Resposta 3"])
    mocker.patch("chat.search_prompt", return_value=chain)
    mocker.patch(
        "builtins.input",
        side_effect=["primeira", "segunda", "terceira", "sair"],
    )

    chat.main()

    captured = capsys.readouterr().out
    assert "RESPOSTA: Resposta 1" in captured
    assert "RESPOSTA: Resposta 2" in captured
    assert "RESPOSTA: Resposta 3" in captured
    assert chain.call_args_list == [call("primeira"), call("segunda"), call("terceira")]
