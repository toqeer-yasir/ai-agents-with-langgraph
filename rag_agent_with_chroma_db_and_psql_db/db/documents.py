from uuid import UUID

async def add_document(
    pool,
    document_id: UUID,
    user_id: UUID,
    filename: str,
    file_path: str,
):
    async with pool.connection() as conn:    
        async with conn.cursor() as cur:

            await cur.execute(
                """
                INSERT INTO documents (
                    id,
                    user_id,
                    filename,
                    file_path
                )
                VALUES (%s, %s, %s, %s);
                """,
                (
                    document_id,
                    user_id,
                    filename,
                    file_path,
                ),
            )



async def get_user_documents(
    pool,
    user_id: UUID,
):
    async with pool.connection() as conn:    
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id,
                    filename,
                    file_path,
                    uploaded_at
                FROM documents
                WHERE user_id = %s
                ORDER BY uploaded_at DESC;
                """,
                (user_id,),
            )

            rows = await cur.fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "file_path": row[2],
                "uploaded_at": row[3],
            }
            for row in rows
        ]


async def get_document(
    pool,
    document_id: UUID,
):
    async with pool.connection() as conn:    
        async with conn.cursor() as cur:

            await cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    filename,
                    file_path,
                    uploaded_at
                FROM documents
                WHERE id = %s;
                """,
                (document_id,),
            )

            row = await cur.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "user_id": row[1],
            "filename": row[2],
            "file_path": row[3],
            "uploaded_at": row[4],
        }


async def delete_document(
    pool,
    document_id: UUID,
):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM documents
                WHERE id = %s
                RETURNING file_name;
                """,
                (document_id,),
            )

            return await cur.fetchone()