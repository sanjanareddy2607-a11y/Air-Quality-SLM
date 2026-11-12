# Pollutants | Air Quality API

Source: https://developers.google.com/maps/documentation/air-quality/pollutants (last updated 2026-09-17 UTC)

## What is a pollutant?

Pollutants are particles and gasses in the air that, in high concentration, can be harmful to people's health.

## Reported pollutants

The Air Quality API supports the most common pollutants.

Pollutants information includes:

- Dominant pollutant
- Concentration
- Sources & Effects

| Code | Display Name | Full Name | Units Measured |
|---|---|---|---|
| co | CO | Carbon monoxide | ppb |
| c6h6 | C6H6 | Benzene | μg/m³ |
| ox | OX | Photochemical oxidants | ppb |
| o3 | O3 | Ozone | ppb |
| nh3 | NH3 | Ammonia | ppb |
| nmhc | NMHC | Non-methane hydrocarbons | ppb |
| no | NO | Nitrogen monoxide | ppb |
| nox | NOX | Nitrogen oxides | ppb |
| no2 | NO2 | Nitrogen dioxide | ppb |
| pm25 | PM2.5 | Fine particulate matter (<2.5µm) | μg/m³ |
| pm10 | PM10 | Inhalable particulate matter (<10µm) | μg/m³ |
| so2 | SO2 | Sulfur dioxide | ppb |
| trs | TRS | Total reduced sulfur | μg/m³ |

*(The PDF text extraction rendered some codes/names imperfectly, e.g. "3" for O3 and "p25"/"P25" for pm25/PM2.5; they are normalized above to match the API's own response examples.)*

---

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2026-09-17 UTC.
