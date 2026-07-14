from uuid import UUID

async def create_user(
    pool,
    user_id: UUID,
    name: str,
    email: str,
):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO users (
                    id,
                    name,
                    email
                )
                VALUES (%s, %s, %s);
                """,
                (
                    user_id,
                    name,
                    email,
                ),
            )


async def get_user_by_id(
    pool,
    user_id: UUID,
):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id,
                    name,
                    email,
                    created_at
                FROM users
                WHERE id = %s;
                """,
                (user_id,),
            )

            row = await cur.fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "created_at": row[3],
    }


async def get_user_by_email(
    pool,
    email: str,
):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    id,
                    name,
                    email,
                    created_at
                FROM users
                WHERE email = %s;
                """,
                (email,),
            )

            row = await cur.fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "created_at": row[3],
    }


async def update_user(
    pool,
    user_id: UUID,
    name: str,
    email: str,
):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE users
                SET
                    name = %s,
                    email = %s
                WHERE id = %s;
                """,
                (
                    name,
                    email,
                    user_id,
                ),
            )

            return cur.rowcount > 0


async def delete_user(
    pool,
    email: str,
):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM users
                WHERE email = %s;
                """,
                (email,),
            )

            return cur.rowcount > 0