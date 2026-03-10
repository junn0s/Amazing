from dotenv import load_dotenv
load_dotenv()

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from image_generate import generate_image
from llm_langchain import MazeState, advance_game

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_state: Optional[MazeState] = None


class MazeResponse(BaseModel):
    width: int
    height: int
    maze: List[List[int]]
    userPos: List[int]
    npcCnt: int
    npcPos: List[List[int]]
    exitPos: List[int]


class MazeRequest(BaseModel):
    loc: List[int]


def get_maze_data() -> MazeResponse:
    return MazeResponse(
        width=11,
        height=11,
        maze=[
            [1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 2, 0, 0, 1, 0, 2, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        userPos=[5, 5],
        npcCnt=3,
        npcPos=[
            [3, 1],
            [3, 6],
            [9, 9],
        ],
        exitPos=[0, 6],
    )


def post_maze_data(req: MazeRequest) -> MazeResponse:
    maze_data = {
        "width": 11,
        "height": 11,
        "maze": [
            [1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 2, 0, 0, 1, 0, 2, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "userPos": [5, 5],
        "npcCnt": 3,
        "npcPos": [
            [3, 1],
            [3, 6],
            [9, 9],
        ],
        "exitPos": [0, 6],
    }

    input_coord = req.loc
    new_npc_pos = [npc for npc in maze_data["npcPos"] if npc != input_coord]
    maze_data["npcPos"] = new_npc_pos
    maze_data["npcCnt"] = len(new_npc_pos)
    maze_data["userPos"] = input_coord

    row, col = input_coord
    maze_data["maze"][row][col] = 3

    return MazeResponse(**maze_data)


class StartRequest(BaseModel):
    name: str
    location: str
    mood: str


class StartResponse(BaseModel):
    worldDescription: str
    image: str


class NpcQuizResponse(BaseModel):
    quiz: str
    option1: str
    option2: str
    option3: str


class NpcQuizResultRequest(BaseModel):
    answer: str


class NpcQuizResultResponse(BaseModel):
    answerDescription: str
    result: int


class EndGameResponse(BaseModel):
    finishDescription: str


@app.post("/maze", response_model=MazeResponse)
def maze_endpoint(req: Optional[MazeRequest] = Body(default=None)):
    if req is None:
        return get_maze_data()
    return post_maze_data(req)


@app.post("/world", response_model=StartResponse)
def start_game(req: StartRequest):
    global game_state
    game_state = MazeState(
        name=req.name,
        setting=req.location,
        atmosphere=req.mood,
        num="0",
        step="start",
        quiz="",
        option1="",
        option2="",
        option3="",
    )

    game_state = advance_game(game_state)

    image_prompt = (
        f"The location is {req.location} and the mood is {req.mood}. "
        "Create a pixel-style image related to this location and mood."
    )
    image_url = generate_image(image_prompt, size="1024x1024")

    return StartResponse(
        worldDescription=game_state.message,
        image=image_url,
    )


@app.get("/npc_quiz", response_model=NpcQuizResponse)
def get_npc_quiz():
    global game_state
    if game_state is None:
        raise HTTPException(status_code=400, detail="게임이 시작되지 않았습니다.")

    valid_quiz_steps = [
        "first_encounter_question",
        "second_encounter_question",
        "third_encounter_question",
    ]
    if game_state.step not in valid_quiz_steps:
        raise HTTPException(
            status_code=400,
            detail=f"현재 {game_state.step} 단계에서는 새 퀴즈를 받을 수 없습니다.",
        )

    game_state = advance_game(game_state)
    return NpcQuizResponse(
        quiz=game_state.quiz,
        option1=game_state.option1,
        option2=game_state.option2,
        option3=game_state.option3,
    )


@app.post("/npc_quiz_result", response_model=NpcQuizResultResponse)
def post_npc_quiz_result(req: NpcQuizResultRequest):
    global game_state
    if game_state is None:
        raise HTTPException(status_code=400, detail="게임이 시작되지 않았습니다.")

    valid_followup_steps = [
        "first_encounter_followup",
        "second_encounter_followup",
        "third_encounter_followup",
    ]
    if game_state.step not in valid_followup_steps:
        raise HTTPException(
            status_code=400,
            detail=f"현재 {game_state.step} 단계에서는 퀴즈 답변을 제출할 수 없습니다.",
        )

    game_state = advance_game(game_state, req.answer)
    return NpcQuizResultResponse(
        answerDescription=game_state.message,
        result=game_state.num,
    )


@app.get("/end_game", response_model=EndGameResponse)
def end_game():
    global game_state
    game_state = advance_game(game_state)
    return EndGameResponse(finishDescription=game_state.message)
