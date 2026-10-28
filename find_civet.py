from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Find Civet Frames', layout='wide')
st.title('Find Frames with Civet')

manifest = Path('data/frames/frames.csv')
outfile = Path('data/civet_frames.csv')

if not manifest.exists():
    st.error('No frames found.')
    st.stop()

frames = pd.read_csv(manifest).fillna('')

# Load existing marks
if outfile.exists():
    existing = pd.read_csv(outfile).fillna('')
    has_civet_map = dict(zip(existing['image_path'], existing['has_civet']))
else:
    has_civet_map = {}

frames['has_civet'] = frames['image_path'].map(has_civet_map).fillna('')

# Stats
total = len(frames)
found = (frames['has_civet'] == 'yes').sum()
not_found = (frames['has_civet'] == 'no').sum()
unmarked = total - found - not_found

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total frames', total)
col2.metric('Has civet', found)
col3.metric('No civet', not_found)
col4.metric('Unmarked', unmarked)

st.divider()

# Controls
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    cam_filter = st.selectbox('Camera', ['All', 'CAM1', 'CAM2', 'CAM3', 'CAM4'])
with col2:
    video_filter = st.selectbox('Video', ['All'] + sorted(frames['source_video'].unique().tolist()))
with col3:
    page_size = st.selectbox('Thumbnails per page', [20, 40, 60], index=1)

# Filter
view = frames.copy()
if cam_filter != 'All':
    view = view[view['camera'] == cam_filter]
if video_filter != 'All':
    view = view[view['source_video'] == video_filter]

# Pagination
if 'page' not in st.session_state:
    st.session_state.page = 0

total_pages = max(1, (len(view) + page_size - 1) // page_size)
st.session_state.page = min(st.session_state.page, total_pages - 1)

start = st.session_state.page * page_size
end = min(start + page_size, len(view))
page_frames = view.iloc[start:end]

st.write(f'Showing frames {start+1}-{end} of {len(view)} | Page {st.session_state.page+1}/{total_pages}')

# Save all marks
def save_marks():
    keep = frames[frames['has_civet'].astype(str).str.strip() != ''][['image_path', 'source_video', 'camera', 'timestamp_seconds', 'has_civet']]
    keep.to_csv(outfile, index=False, encoding='utf-8-sig')

# Show thumbnails in grid
cols_per_row = 5
for i in range(0, len(page_frames), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < len(page_frames):
            frame = page_frames.iloc[i + j]
            with col:
                try:
                    img = Image.open(frame['image_path'])
                    img.thumbnail((200, 200))
                    st.image(img, use_container_width=True)
                except:
                    st.error('Error')

                # Show current mark
                mark = frame['has_civet']
                if mark == 'yes':
                    st.success('CIVET')
                elif mark == 'no':
                    st.info('Empty')

                # Buttons
                b1, b2 = st.columns(2)
                if b1.button('YES', key=f"yes_{frame['image_path']}", type='primary'):
                    frames.loc[frames['image_path'] == frame['image_path'], 'has_civet'] = 'yes'
                    save_marks()
                    st.rerun()
                if b2.button('NO', key=f"no_{frame['image_path']}"):
                    frames.loc[frames['image_path'] == frame['image_path'], 'has_civet'] = 'no'
                    save_marks()
                    st.rerun()

# Pagination buttons
st.divider()
p1, p2, p3 = st.columns([1, 2, 1])
if p1.button('← Previous', disabled=st.session_state.page == 0):
    st.session_state.page -= 1
    st.rerun()
p2.write(f'Page {st.session_state.page + 1} of {total_pages}')
if p3.button('Next →', disabled=st.session_state.page >= total_pages - 1):
    st.session_state.page += 1
    st.rerun()
