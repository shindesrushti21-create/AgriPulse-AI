// ============================================
// AgriPulse AI - Week 1: Monthly Data Pipeline
// District: Kolhapur - FULL RANGE 2019-2026
// ============================================


var districts = ee.FeatureCollection('FAO/GAUL/2015/level2');
var studyArea = districts
  .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Kolhapur'));

Map.centerObject(studyArea, 8);
Map.addLayer(studyArea, {color: 'red'}, 'Kolhapur District Boundary');
var geometry = studyArea.geometry();

var startYear = 2019;
var endYear = 2026;
var months = ee.List.sequence(1, 12);
var years = ee.List.sequence(startYear, endYear);

var now = ee.Date(Date.now());
var currentMonthStart = ee.Date.fromYMD(now.get('year'), now.get('month'), 1);

var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geometry);

function maskCloudsQA60(img) {
  var qa = img.select('QA60');
  var cloudBit = 1 << 10;
  var cirrusBit = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBit).eq(0).and(qa.bitwiseAnd(cirrusBit).eq(0));
  var masked = img.updateMask(mask);
  var ndvi = masked.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndmi = masked.normalizedDifference(['B8', 'B11']).rename('NDMI');
  return masked.addBands(ndvi).addBands(ndmi);
}

var s2Clean = s2Sr.map(maskCloudsQA60);
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(geometry);
var modisLST = ee.ImageCollection('MODIS/061/MOD11A2').filterBounds(geometry);

// Helper: safely reduce a collection, returns a masked placeholder if empty
function safeBand(collection, bandName) {
  var count = collection.size();
  var placeholder = ee.Image(0).rename(bandName).selfMask();
  return ee.Image(ee.Algorithms.If(count.gt(0), collection.mean().rename(bandName), placeholder));
}

var monthlyList = years.map(function(y) {
  return months.map(function(m) {
    y = ee.Number(y);
    m = ee.Number(m);
    var start = ee.Date.fromYMD(y, m, 1);
    var end = start.advance(1, 'month');
    var isComplete = end.millis().lte(currentMonthStart.millis());

    var s2Monthly = s2Clean.filterDate(start, end);
    var imageCount = s2Monthly.size();

    var ndviImg = safeBand(s2Monthly.select('NDVI'), 'NDVI');
    var ndmiImg = safeBand(s2Monthly.select('NDMI'), 'NDMI');

    var rainCollection = chirps.filterDate(start, end);
    var rainCount = rainCollection.size();
    var rainImg = ee.Image(ee.Algorithms.If(
      rainCount.gt(0),
      rainCollection.sum().rename('rainfall_mm'),
      ee.Image(0).rename('rainfall_mm').selfMask()
    ));

    var lstCollection = modisLST.filterDate(start, end).select('LST_Day_1km');
    var lstRaw = safeBand(lstCollection, 'LST_Day_1km');
    var lstImg = lstRaw.multiply(0.02).subtract(273.15).rename('LST_celsius');

    var combined = ndviImg.addBands(ndmiImg).addBands(rainImg).addBands(lstImg);

    var stats = combined.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geometry,
      scale: 100,
      maxPixels: 1e9,
      tileScale: 8,
      bestEffort: true
    });

    return ee.Feature(null, {
      'year': y,
      'month': m,
      'date': start.format('YYYY-MM'),
      'district': 'Kolhapur',
      'NDVI': stats.get('NDVI'),
      'NDMI': stats.get('NDMI'),
      'rainfall_mm': stats.get('rainfall_mm'),
      'LST_celsius': stats.get('LST_celsius'),
      'sentinel_image_count': imageCount,
      'is_complete_month': isComplete
    });
  });
}).flatten();

var monthlyTable = ee.FeatureCollection(monthlyList)
  .filter(ee.Filter.eq('is_complete_month', 1));

Export.table.toDrive({
  collection: monthlyTable,
  description: 'kolhapur_monthly_features_2019_2026',
  fileFormat: 'CSV'
});

print('Script loaded. Go to Tasks tab and click Run on the export.');