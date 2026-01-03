from config import API_ID, API_HASH
from telethon import TelegramClient
from telethon.errors import (
    ApiIdInvalidError,
    AuthKeyUnregisteredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    FloodWaitError,
    RPCError
)
from telethon.tl.custom import Message
import sqlite3
import asyncio
from datetime import datetime, timezone


api_id = API_ID
api_hash = API_HASH

session_name = "scrapper_session"
channel_username = "cubadebate"

DB_PATH = "noticias.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# rango de fechas deseado
start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 10, 31, 23, 59, 59, tzinfo=timezone.utc)

cursor.execute('''
    CREATE TABLE IF NOT EXISTS noticias (
        id INTEGER PRIMARY KEY,
        msg_id INTEGER UNIQUE,
        fecha TEXT,
        texto TEXT
    )
'''
)

conn.commit()

async def main():
    try:
        # crear cliente
        client = TelegramClient(session=session_name, api_id=api_id, api_hash=api_hash)

    except ApiIdInvalidError:
        print('Error: api_id o api_hash invalidos')

    except PhoneNumberInvalidError:
        print('ERROR: número de teléfono no válido para esta cuenta de Telegram')

    except PhoneCodeInvalidError:
        print("ERROR: código de verificación incorrecto.")

    except AuthKeyUnregisteredError:
        print("ERROR: la sesión guardada ya no es válida. Borra el archivo .session e intenta de nuevo.")

    except FloodWaitError as e:
        print(f"Has hecho demasiadas peticiones. Debes esperar {e.seconds} segundos antes de reintentar.")

    except RPCError as e:
        # Cubre otros errores de la API de Telegram
        print(f"Error de Telegram API: {e}")

    except Exception as e:
        # Cubre cualquier otro error inesperado
        print(f"Error inesperado: {e}")

    async with client:
        try:
            me = await client.get_me()
            print(f"Conectado como: {me.username or me.first_name}")
        except RPCError as e:
            print(f"Error RPC al conectar: {e}")
            return
                
        total = 0
        primer_mensaje = None

        try:
            # reverse=True hace que empiece por el mas antiguo
            async for message in client.iter_messages(channel_username, limit=None):
                if not isinstance(message, Message) or not message.text:
                    continue

                if (total % 1000 == 0):
                    print(total)

                msg_date = message.date # datetime
                
                if msg_date > end_date:
                    continue

                if msg_date < start_date:
                    print("Ya se llego hasta el inicio de 2023, deteniendo iteracion ...")
                    break

                if start_date <= msg_date <= end_date:
                    total += 1
                    if (total % 1000 == 0):
                        print(total)

                    msg_id = message.id
                    fecha = msg_date.isoformat()
                    texto = message.text

                    try:
                        cursor.execute('''
                            INSERT OR IGNORE INTO noticias (msg_id, fecha, texto)
                            VALUES (?, ?, ?)
                        ''', (msg_id, fecha, texto)
                        )

                        conn.commit()
                    
                    except sqlite3.Error as db_err:
                        print(f"Error en la base de datos: {db_err}")



                if primer_mensaje == None:
                    primer_mensaje = message
                    print("=== Primer mensaje del canal ===")
                    print(f"ID: {primer_mensaje.id}")
                    print(f"Fecha: {primer_mensaje.date}")
                    print(f"Remitente (sender_id): {primer_mensaje.sender_id}")
                    print(f"Texto:\n{primer_mensaje.text}")

            print(f"total de mensajes entre 2023 y 2024 insertados: {total}")
                    
        except FloodWaitError as e:
            print(f"Has hecho demasiadas peticiones. Espera {e.seconds}")
        

        



if __name__ == "__main__":
    asyncio.run(main())