# Heatmaps | Air Quality API

Source: https://developers.google.com/maps/documentation/air-quality/heatmaps (last updated 2026-09-17 UTC)

The [heatmapTiles](/maps/documentation/air-quality/reference/rest/v1/mapTypes.heatmapTiles/lookupHeatmapTile) endpoint provides air quality related heatmaps through a tile overlay. A tile overlay is a collection of PNG image tiles that can be displayed on top of a Google Map. Each image tile is 256 x 256 pixels.

## About heatmap tiles

The tile overlay is a grid assembled from a collection of tiles, each assigned (X,Y) coordinates. The tile with coordinates (0,0) is always at the northwest corner of the map. The X values increase from west to east, while the Y values increase from north to south.

The zoom level determines how large or small the contents of a map appear in a map view. The tile grid size is determined by scaling the X and Y coordinates exponentially by the zoom level.

```
gridSizeX = X^zoomLevel
gridSizeY = Y^zoomLevel
```

*(As printed in the source; the grid for zoom level z is 2^z × 2^z tiles, per the example below.)*

For example, at a zoom level of 2, the world map is represented using a 4 x 4 matrix, for a total of 16 tiles. The zoom also determines what is the max value allowed for the coordinates.

The following image shows a US_AQI heatmap with a zoom level of 2, along with the coordinates of each tile:

*[Figure: 4×4 tile grid of a US_AQI heatmap at zoom level 2, with tiles labelled 0,0 through 3,3.]*

| | x=0 | x=1 | x=2 | x=3 |
|---|---|---|---|---|
| **y=0** | 0,0 | 1,0 | 2,0 | 3,0 |
| **y=1** | 0,1 | 1,1 | 2,1 | 3,1 |
| **y=2** | 0,2 | 1,2 | 2,2 | 3,2 |
| **y=3** | 0,3 | 1,3 | 2,3 | 3,3 |

See [Map and Tile coordinates](/maps/documentation/javascript/coordinates) for additional details.

## About the heatmap endpoint

A heatmap tile is represented by a bytes array containing the tile data as a PNG image. You request current air quality heatmap tiles using the `heatmapTiles` endpoint by sending an HTTP GET request to:

```
https://airquality.googleapis.com/v1/mapTypes/TYPE/heatmapTiles/Z/X/Y?key=YOUR_API_KEY
```

All request parameters are required in a request:

- **TYPE** - The type of heatmap to return. See [allowed values](/maps/documentation/air-quality/reference/rest/v1/mapTypes.heatmapTiles/lookupHeatmapTile#maptype).
- **Z** - Zoom level that determines how large or small the contents of a map appear in a map view. Allowed values are 0-16, where a value of 0 displays the entire world in a single tile.
- **X,Y** - The coordinates of the tile to retrieve, relative to the north west corner (0,0). X values are increasing from west to east and Y values are increasing from north to south.

  Tile coordinates have to be valid for the specified zoom level. For example, if you set the zoom level to 2, and request a tile at coordinates of 10,10, the API returns an error.
- **YOUR_API_KEY** - Your application's API key. This key identifies your application for purposes of quota management. For more information, see [get a key](/maps/documentation/air-quality/get-api-key).

## Example heatmap request

The image above shows a `US_AQI` heatmap with a zoom level of 2, along with the coordinates of each tile. The following example uses the `heatmapTiles` endpoint to request the tile at coordinates 0,1 from the heatmap:

```
https://airquality.googleapis.com/v1/mapTypes/US_AQI/heatmapTiles/2/0/1?key=YOUR_API_KEY
```

The tile appears as:

*[Figure: heatmap tile covering western North America (Alaska to Mexico), colored green to red/purple by AQI.]*

For more information on overlaying tiles on top of the default map, see [Overlay Map Types](/maps/documentation/javascript/examples/maptype-overlay).

---

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2026-09-17 UTC.
