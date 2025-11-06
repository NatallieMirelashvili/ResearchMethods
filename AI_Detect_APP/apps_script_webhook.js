const SHEET_ID = '1TH4GpNIiv0w_CSyPs-yJNeEJwacHVLZLSlWmLOYtnJA';
const SECRET   = 'PUT_A_RANDOM_SECRET_HERE';

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents || '{}');
    if (data.secret !== SECRET) {
      return ContentService.createTextOutput('forbidden').setMimeType(ContentService.MimeType.TEXT);
    }
    const ss = SpreadsheetApp.openById(SHEET_ID);
    const ws = ss.getSheetByName('responses') || ss.insertSheet('responses');
    if (ws.getLastRow() === 0) {
      ws.appendRow(['timestamp','participant_email','pair_id','left_url','right_url','left_type','right_type','choice','result']);
    }
    ws.appendRow([
      new Date(),
      data.participant_id || '',
      data.pair_id || '',
      data.left_url || '',
      data.right_url || '',
      data.left_type || '',
      data.right_type || '',
      data.choice || '',
      (data.result === 'T' ? 'T' : 'F')
    ]);
    return ContentService.createTextOutput('OK').setMimeType(ContentService.MimeType.TEXT);
  } catch (err) {
    return ContentService.createTextOutput('ERR:' + err).setMimeType(ContentService.MimeType.TEXT);
  }
}
