from functools import lru_cache
from pathlib import Path
import re

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

app = FastAPI(name="bookapi", title="Le Wagon Book API")


class BookRecord(BaseModel):
    isbn13: int
    authors: str
    title: str
    num_pages: int


class BookRecordWithCover(BookRecord):
    cover_url: str


@lru_cache
def _get_books() -> dict[int, BookRecord]:
    FP_BOOK = Path(__file__).parent / "data/books.csv"
    books_df = pd.read_csv(FP_BOOK, on_bad_lines="skip")
    books_df = books_df.drop(
        columns=[
            "bookID",
            "isbn",
            "average_rating",
            "language_code",
            "ratings_count",
            "text_reviews_count",
        ]
    ).rename(columns={"# num_pages": "num_pages"})
    books_list: list[dict] = books_df.to_dict(orient="records")
    return {val["isbn13"]: BookRecord(**val) for val in books_list}


DB = _get_books()


@app.get("/")
def hello():
    return {"Hello": "World"}


@app.get("/books/{isbn}")
def book(isbn: int, request: Request) -> BookRecordWithCover:
    book = DB.get(isbn)
    if book is None:
        raise HTTPException(status_code=404, detail=f"Book not found with isbn {isbn}")
    base_url = re.findall(r"http://([^/]+)/", str(request.url))[0]
    cover_url = f"http://{base_url}/covers/{book.isbn13}.png"
    return BookRecordWithCover(**book.model_dump(), cover_url=cover_url)


@app.get(
    "/covers/{image_path}",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
def get_cover(image_path: str) -> Response:
    """Returns random covers PNG covers"""
    if image_path[-4:] != ".png":
        raise HTTPException(status_code=404, detail="Only png files available")
    isbn_str = image_path[:-4]
    try:
        isbn = int(isbn_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image path")
    book = DB.get(isbn)
    if book is None:
        raise HTTPException(status_code=404, detail=f"Book not found with isbn {isbn}")
    image_bytes: bytes = b"aaffda"
    return Response(content=image_bytes, media_type="image/png")
