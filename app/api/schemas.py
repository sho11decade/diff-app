from pydantic import BaseModel


class DifferencePosition(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DifferenceCard(BaseModel):
    index: int
    edit_type: str
    edit_strength: float
    region_area: int
    difficulty_factor: float
    score_breakdown: dict[str, float]


class GenerateResponse(BaseModel):
    puzzle_image_base64: str
    answer_image_base64: str
    positions: list[DifferencePosition]
    processing_time_ms: int
    trace_id: str | None = None
    seed: int | None = None
    difference_cards: list[DifferenceCard] | None = None
    trace_log_path: str | None = None
