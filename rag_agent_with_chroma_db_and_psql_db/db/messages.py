from uuid import UUID

async def create_message(pool, id, chat_id, user_content, assistant_content):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO messages(id, chat_id, role, content)
                VALUES (%s, %s, 'user', %s)
                """,
                (id, chat_id, user_content)
            )

            user_message_id = (await cur.fetchone()[0])

            await cur.execute(
                """
                INSERT INTO messages(id, chat_id, role, content)
                VALUES (%s, %s, 'assistant', %s)
                """,
                (id, chat_id, assistant_content, user_message_id)
            )

            


async def get_chat_messages(pool, chat_id):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:

            await cur.execute(
                """
                SELECT
                id,
                role,
                content,
                parent_message_id,
                created_at
                FROM messages
                WHERE chat_id = %s
                ORDER BY created_at ASC;
                """,
                (chat_id,),
            )

            rows = await cur.fetchall()

        return [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "parent_message_id": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]