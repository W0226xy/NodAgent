import json
import logging
from datetime import (
    datetime,
)
from zoneinfo import (
    ZoneInfo,
)

import httpx
from fastmcp import FastMCP


logger = logging.getLogger(
    __name__
)


mcp = FastMCP(
    "NodAgent External Info Server"
)


@mcp.tool()
def get_current_time(
    timezone: str,
) -> str:
    """
    Get the current date and time for an
    IANA timezone.

    Example:
    Asia/Shanghai
    Asia/Tokyo
    America/New_York
    """

    try:
        now = datetime.now(
            ZoneInfo(timezone)
        )

    except Exception:
        return json.dumps(
            {
                "error": (
                    "Invalid timezone"
                ),
                "timezone": timezone,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "timezone": timezone,
            "datetime": (
                now.isoformat()
            ),
        },
        ensure_ascii=False,
)


def weather_error(
    code: str,
    message: str,
    city: str,
) -> str:
    return json.dumps(
        {
            "error": {
                "code": code,
                "message": message,
            },
            "city": city,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def get_weather(
    city: str,
) -> str:
    """
    Get current weather for a city.

    This tool uses the public Open-Meteo
    geocoding and weather APIs.
    """

    city = city.strip()

    if not city:
        return weather_error(
            code="invalid_city",
            message="请提供要查询的城市。",
            city=city,
        )

    timeout = httpx.Timeout(
        10.0
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            geo_response = await client.get(
                (
                    "https://geocoding-api."
                    "open-meteo.com/v1/search"
                ),
                params={
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )

            geo_response.raise_for_status()

            geo_data = geo_response.json()

            if not isinstance(
                geo_data,
                dict,
            ):
                raise ValueError(
                    "Invalid geocoding response"
                )

            results = geo_data.get(
                "results",
                [],
            )

            if not isinstance(
                results,
                list,
            ):
                raise ValueError(
                    "Invalid geocoding results"
                )

            if not results:
                return weather_error(
                    code="city_not_found",
                    message=(
                        "未找到该城市，请检查城市名称后重试。"
                    ),
                    city=city,
                )

            location = results[0]

            if not isinstance(
                location,
                dict,
            ):
                raise ValueError(
                    "Invalid location result"
                )

            latitude = location.get(
                "latitude"
            )
            longitude = location.get(
                "longitude"
            )

            if not isinstance(
                latitude,
                (int, float),
            ) or not isinstance(
                longitude,
                (int, float),
            ):
                raise ValueError(
                    "Invalid location coordinates"
                )

            weather_response = await client.get(
                (
                    "https://api."
                    "open-meteo.com/"
                    "v1/forecast"
                ),
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,"
                        "apparent_temperature,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )

            weather_response.raise_for_status()

            weather_data = weather_response.json()

            if not isinstance(
                weather_data,
                dict,
            ) or not isinstance(
                weather_data.get(
                    "current"
                ),
                dict,
            ):
                raise ValueError(
                    "Invalid weather response"
                )

    except httpx.TimeoutException:
        return weather_error(
            code="weather_service_timeout",
            message=(
                "天气服务连接超时，请稍后重试。"
            ),
            city=city,
        )

    except httpx.HTTPStatusError:
        return weather_error(
            code="weather_service_http_error",
            message=(
                "天气服务暂时不可访问，请稍后重试。"
            ),
            city=city,
        )

    except httpx.RequestError as exc:
        logger.warning(
            "Weather request failed (%s)",
            type(exc).__name__,
        )
        return weather_error(
            code="weather_service_unavailable",
            message=(
                "天气服务暂时不可访问，请稍后重试。"
            ),
            city=city,
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "Weather response was invalid (%s)",
            type(exc).__name__,
        )
        return weather_error(
            code="weather_service_invalid_response",
            message=(
                "天气服务返回异常，请稍后重试。"
            ),
            city=city,
        )

    except Exception as exc:
        logger.warning(
            "Weather lookup failed (%s)",
            type(exc).__name__,
        )
        return weather_error(
            code="weather_service_error",
            message=(
                "天气服务暂时不可访问，请稍后重试。"
            ),
            city=city,
        )

    return json.dumps(
        {
            "city": location.get(
                "name"
            ),
            "country": location.get(
                "country"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": (
                weather_data.get(
                    "timezone"
                )
            ),
            "current": (
                weather_data.get(
                    "current"
                )
            ),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )
