from dotenv import load_dotenv
load_dotenv()

import json
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

from langchain.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class MazeState(BaseModel):
    name: str
    setting: str
    atmosphere: str

    quiz: str
    option1: str
    option2: str
    option3: str

    num: str

    step: str = "start"
    message: str = ""

    inventory: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    story_data: Optional[dict] = None
    player_answer: str = ""


class GraphState(TypedDict, total=False):
    name: str
    setting: str
    atmosphere: str
    quiz: str
    option1: str
    option2: str
    option3: str
    num: str
    step: str
    message: str
    inventory: List[str]
    history: List[str]
    story_data: Optional[Dict[str, Any]]
    player_answer: str


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
)


QUESTION_CONFIG = {
    "first_encounter_question": {
        "npc_index": 0,
        "story_key": "intro",
        "next_step": "first_encounter_followup",
    },
    "second_encounter_question": {
        "npc_index": 1,
        "story_key": "middle",
        "next_step": "second_encounter_followup",
    },
    "third_encounter_question": {
        "npc_index": 2,
        "story_key": "final",
        "next_step": "third_encounter_followup",
    },
}


FOLLOWUP_CONFIG = {
    "first_encounter_followup": {
        "npc_index": 0,
        "next_step": "second_encounter_question",
    },
    "second_encounter_followup": {
        "npc_index": 1,
        "next_step": "third_encounter_question",
    },
    "third_encounter_followup": {
        "npc_index": 2,
        "next_step": "end_game",
    },
}


def clean_response(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _dump_state(state: MazeState) -> GraphState:
    return cast(GraphState, state.model_dump())


def _load_state(state: GraphState) -> MazeState:
    return MazeState.model_validate(state)


def _invoke_json(prompt: str) -> Tuple[Dict[str, Any], str]:
    raw_response = llm.invoke(prompt).content
    cleaned_response = clean_response(raw_response)
    return json.loads(cleaned_response), raw_response


def _require_story_data(state: MazeState) -> Dict[str, Any]:
    if state.story_data is None:
        raise ValueError("story_data가 없습니다.")
    return cast(Dict[str, Any], state.story_data)


def _generate_world_data(state: MazeState) -> MazeState:
    prompt = f"""
    게임 개요: 사용자가 설정한 장소와 분위기 기반으로 AI가 세계관과 스토리를 생성하며, NPC와의 상호작용이 중요한 요소입니다.
    플레이어가 미로의 장소로 '{state.setting}', 분위기로 '{state.atmosphere}'를 입력했습니다.
    플레이어의 이름은 {state.name} 입니다.

    위 입력을 바탕으로 방탈출게임 느낌의 세계관, 목표, 주요 스토리 흐름만 생성해주세요.
    이 단계에서는 NPC 정보는 만들지 마세요.
    아래 예시와 동일한 순수 JSON만 출력하고, 삼중 백틱은 사용하지 마세요.

    {{
        "world_description": "세계관 및 스토리 전체 내용",
        "objective": "미로탈출 게임의 목표를 존댓말로 설명",
        "story_details": {{
            "background": "스토리 시작 설명",
            "intro": "초반부 진행 상황 설명",
            "middle": "중반부 진행 상황 설명",
            "final": "후반부 진행 상황 설명",
            "result": "최종 결말 설명"
        }}
    }}
    """
    try:
        story_data, _ = _invoke_json(prompt)
        state.story_data = story_data
    except json.JSONDecodeError:
        state.message = "생성에 실패했습니다."
    return state


def _generate_npc_roster(state: MazeState) -> MazeState:
    try:
        story_data = _require_story_data(state)
    except ValueError:
        state.message = "스토리 정보가 없어 NPC를 생성할 수 없습니다."
        return state

    story_details = json.dumps(story_data.get("story_details", {}), ensure_ascii=False)
    prompt = f"""
    당신은 게임 기획자입니다.
    아래 세계관과 스토리 흐름을 바탕으로 NPC 3명을 생성해주세요.

    세계관: {story_data.get("world_description", "")}
    목표: {story_data.get("objective", "")}
    스토리 흐름: {story_details}

    각 NPC는 이름, 직업, 말투/성격을 가져야 하며 세계관에 자연스럽게 어울려야 합니다.
    아래 형식과 동일한 순수 JSON만 출력하고, 삼중 백틱은 사용하지 마세요.

    {{
        "npcs": [
            {{"name": "NPC 이름", "role": "NPC 직업", "personality": "성격과 말투"}},
            {{"name": "NPC 이름", "role": "NPC 직업", "personality": "성격과 말투"}},
            {{"name": "NPC 이름", "role": "NPC 직업", "personality": "성격과 말투"}}
        ]
    }}
    """
    try:
        npc_data, _ = _invoke_json(prompt)
        story_data["npcs"] = npc_data["npcs"]
        state.story_data = story_data
    except (json.JSONDecodeError, KeyError):
        state.message = "NPC 생성에 실패했습니다."
    return state


def _build_opening_message(state: MazeState) -> MazeState:
    try:
        story_data = _require_story_data(state)
    except ValueError:
        state.message = "스토리 정보가 없습니다."
        return state

    background = story_data.get("story_details", {}).get("background", "")
    objective = story_data.get("objective", "")
    state.message = background + "\n" + objective + "\n행운을 빕니다!\n"
    state.step = "first_encounter_question"
    return state


def _generate_question(state: MazeState, step_name: str) -> MazeState:
    config = QUESTION_CONFIG[step_name]
    try:
        story_data = _require_story_data(state)
    except ValueError:
        state.message = "스토리 정보가 없습니다."
        return state

    npc = story_data["npcs"][config["npc_index"]]
    story_segment = story_data.get("story_details", {}).get(config["story_key"], "")

    prompt = f"""
    당신은 이 미로 속에서 플레이어가 만나는 NPC '{npc['name']}' (직업: {npc['role']}) 입니다.
    당신의 말투는 {npc.get('personality', '')} 입니다.
    먼저 '{story_segment}'를 말하고, 그 내용에 기반한 객관식 3지선다 퀴즈를 1개 내주세요.
    틀리면 패널티가 있다는 말도 포함해주세요.

    아래 형식과 동일한 순수 JSON만 출력하고, 삼중 백틱은 사용하지 마세요.
    {{
        "quiz": "스토리 설명과 퀴즈 질문",
        "option1": "1번 선택지",
        "option2": "2번 선택지",
        "option3": "3번 선택지"
    }}
    """
    try:
        data, raw_response = _invoke_json(prompt)
        state.quiz = data["quiz"]
        state.option1 = data["option1"]
        state.option2 = data["option2"]
        state.option3 = data["option3"]
        state.message = state.quiz
        state.history.append(f"{npc['name']}: {raw_response}")
    except (json.JSONDecodeError, KeyError):
        state.message = "퀴즈 생성에 실패했습니다. 다시 시도해주세요."

    state.step = config["next_step"]
    return state


def _generate_followup(state: MazeState, step_name: str) -> MazeState:
    config = FOLLOWUP_CONFIG[step_name]
    try:
        story_data = _require_story_data(state)
    except ValueError:
        state.message = "스토리 정보가 없습니다."
        return state

    npc = story_data["npcs"][config["npc_index"]]
    player_answer = state.player_answer.strip()
    quiz_context = json.dumps(
        {
            "quiz": state.quiz,
            "option1": state.option1,
            "option2": state.option2,
            "option3": state.option3,
        },
        ensure_ascii=False,
    )

    prompt = f"""
    당신은 NPC '{npc['name']}' 입니다. 당신의 말투는 {npc.get('personality', '')} 입니다.
    플레이어가 '{player_answer}' 라고 답했습니다.
    이전 퀴즈 정보는 {quiz_context} 입니다.

    아래 형식과 동일한 순수 JSON만 출력하고, 삼중 백틱은 사용하지 마세요.
    {{
        "message": "플레이어의 답이 맞으면 정답이라고 말하고 자연스러운 대화를 한 마디 추가하세요. 틀리면 틀렸다고 말하고 자연스러운 대화를 한 마디 추가하세요.",
        "answer": "정답이면 0, 오답이면 1"
    }}
    """
    try:
        data, raw_response = _invoke_json(prompt)
        state.message = data["message"]
        state.num = str(data["answer"])
        state.history.append(f"플레이어: {player_answer}")
        state.history.append(f"{npc['name']}: {raw_response}")
    except (json.JSONDecodeError, KeyError):
        state.message = "결과 로드에 실패했습니다. 다시 시도해주세요."

    state.step = config["next_step"]
    return state


def _generate_ending(state: MazeState) -> MazeState:
    try:
        story_data = _require_story_data(state)
    except ValueError:
        state.message = "스토리 정보가 없습니다."
        return state

    result_story = story_data.get("story_details", {}).get("result", "")
    prompt = f"""
    미로의 마지막 장소에 도착했습니다.
    스토리의 최종 결말인 '{result_story}'를 플레이어에게 자세하게 설명해주세요.
    """
    state.message = llm.invoke(prompt).content
    state.step = "game_finished"
    return state


def _game_finished(state: MazeState) -> MazeState:
    state.message = "게임이 이미 종료되었습니다."
    return state


def _invalid_step(state: MazeState) -> MazeState:
    state.message = f"알 수 없는 단계: {state.step}"
    return state


def _generate_world_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_world_data(_load_state(state)))


def _generate_npc_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_npc_roster(_load_state(state)))


def _opening_node(state: GraphState) -> GraphState:
    return _dump_state(_build_opening_message(_load_state(state)))


def _first_question_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_question(_load_state(state), "first_encounter_question"))


def _second_question_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_question(_load_state(state), "second_encounter_question"))


def _third_question_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_question(_load_state(state), "third_encounter_question"))


def _first_followup_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_followup(_load_state(state), "first_encounter_followup"))


def _second_followup_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_followup(_load_state(state), "second_encounter_followup"))


def _third_followup_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_followup(_load_state(state), "third_encounter_followup"))


def _ending_node(state: GraphState) -> GraphState:
    return _dump_state(_generate_ending(_load_state(state)))


def _finished_node(state: GraphState) -> GraphState:
    return _dump_state(_game_finished(_load_state(state)))


def _invalid_step_node(state: GraphState) -> GraphState:
    return _dump_state(_invalid_step(_load_state(state)))


def _route_current_step(state: GraphState) -> str:
    step = state.get("step", "start")
    routes = {
        "start": "generate_world",
        "first_encounter_question": "first_encounter_question",
        "first_encounter_followup": "first_encounter_followup",
        "second_encounter_question": "second_encounter_question",
        "second_encounter_followup": "second_encounter_followup",
        "third_encounter_question": "third_encounter_question",
        "third_encounter_followup": "third_encounter_followup",
        "end_game": "end_game",
        "game_finished": "game_finished",
    }
    return routes.get(step, "invalid_step")


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("generate_world", _generate_world_node)
    graph.add_node("generate_npcs", _generate_npc_node)
    graph.add_node("opening_message", _opening_node)
    graph.add_node("first_encounter_question", _first_question_node)
    graph.add_node("first_encounter_followup", _first_followup_node)
    graph.add_node("second_encounter_question", _second_question_node)
    graph.add_node("second_encounter_followup", _second_followup_node)
    graph.add_node("third_encounter_question", _third_question_node)
    graph.add_node("third_encounter_followup", _third_followup_node)
    graph.add_node("end_game", _ending_node)
    graph.add_node("game_finished", _finished_node)
    graph.add_node("invalid_step", _invalid_step_node)

    graph.add_conditional_edges(
        START,
        _route_current_step,
        {
            "generate_world": "generate_world",
            "first_encounter_question": "first_encounter_question",
            "first_encounter_followup": "first_encounter_followup",
            "second_encounter_question": "second_encounter_question",
            "second_encounter_followup": "second_encounter_followup",
            "third_encounter_question": "third_encounter_question",
            "third_encounter_followup": "third_encounter_followup",
            "end_game": "end_game",
            "game_finished": "game_finished",
            "invalid_step": "invalid_step",
        },
    )

    graph.add_edge("generate_world", "generate_npcs")
    graph.add_edge("generate_npcs", "opening_message")
    graph.add_edge("opening_message", END)
    graph.add_edge("first_encounter_question", END)
    graph.add_edge("first_encounter_followup", END)
    graph.add_edge("second_encounter_question", END)
    graph.add_edge("second_encounter_followup", END)
    graph.add_edge("third_encounter_question", END)
    graph.add_edge("third_encounter_followup", END)
    graph.add_edge("end_game", END)
    graph.add_edge("game_finished", END)
    graph.add_edge("invalid_step", END)

    return graph.compile()


game_graph = _build_graph()


def advance_game(state: MazeState, player_answer: Optional[str] = None) -> MazeState:
    if player_answer is not None:
        state.player_answer = player_answer.strip()

    next_state = game_graph.invoke(_dump_state(state))
    return MazeState.model_validate(next_state)
