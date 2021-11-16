from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey, insert, select
from sqlalchemy.orm import Session


class Database:
    def __init__(self):
        self.engine = create_engine(
            "postgresql://admin_db:admin_12345678@postgresql:5432/postgres_db"
        )
        self.metadata = MetaData()

        self.languages_table = Table(
            "languages",
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, nullable=False, unique=True))

        self.corrections_table = Table(
            "corrections",
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('from_lang', ForeignKey("languages.id"), nullable=False),
            Column('to_lang', ForeignKey("languages.id"), nullable=False),
            Column('text', String),
            Column('translation', String),
            Column('translation_correction', String))

        self.metadata.create_all(self.engine)

    def save_correction(self, from_lang, to_lang, text, translation, translation_correction):
        lang1 = self.get_or_create_language(from_lang)
        lang2 = self.get_or_create_language(to_lang)
        stmt = insert(self.corrections_table).values(
            from_lang=lang1,
            to_lang=lang2,
            text=text,
            translation=translation,
            translation_correction=translation_correction
        )
        with Session(self.engine) as session:
            session.execute(stmt)
            session.commit()

    def get_or_create_language(self, language: str) -> int:
        check_query = select(self.languages_table).where(
            self.languages_table.c.name == language)
        insert_query = insert(self.languages_table).values(
            name=language
        )
        with Session(self.engine) as session:
            result = session.execute(check_query)
            data = result.first()
            if data is None:
                result = session.execute(insert_query)
                session.commit()
                data = result.inserted_primary_key
            return data[0]
