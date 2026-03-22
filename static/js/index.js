window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  bulmaSlider.attach();

  // =====================================================================
  //  GALLERY DATA — Add your editing cases here!
  //  Each item: { category, instruction, sourceImg, video }
  //  - category: one of the edit types for filtering
  //  - instruction: the text editing instruction
  //  - sourceImg: path to source 3D render (front view)
  //  - video: path to 360° turntable video of the edited result
  //
  //  To add a new case, just append an object to this array.
  // =====================================================================
  const galleryData = [
    {
      category: "Addition",
      instruction: "Add a ladder ascending on the left side of the structure.",
      sourceVideo: "./assets/videos/cases/all6/all6-01-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-01-edited.png",
      video: "./assets/videos/cases/all6/all6-01.mp4",
    },
    {
      category: "Addition",
      instruction: "Add white polka dots to the cap.",
      sourceVideo: "./assets/videos/cases/all6/all6-02-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-02-edited.png",
      video: "./assets/videos/cases/all6/all6-02.mp4",
    },
    {
      category: "Addition",
      instruction: "Include a small bronze spigot on the front side of the barrel.",
      sourceVideo: "./assets/videos/cases/all6/all6-05-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-05-edited.png",
      video: "./assets/videos/cases/all6/all6-05.mp4",
    },
    {
      category: "Removal",
      instruction: "Remove the irregular stones from around the base of the mug.",
      sourceVideo: "./assets/videos/cases/all6/all6-07-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-07-edited.png",
      video: "./assets/videos/cases/all6/all6-07.mp4",
    },
    {
      category: "Removal",
      instruction: "Remove the black padding visible inside the heel of the shoe.",
      sourceVideo: "./assets/videos/cases/all6/all6-08-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-08-edited.png",
      video: "./assets/videos/cases/all6/all6-08.mp4",
    },
    {
      category: "Removal",
      instruction: "Remove the cap from the pouch.",
      sourceVideo: "./assets/videos/cases/all6/all6-12-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-12-edited.png",
      video: "./assets/videos/cases/all6/all6-12.mp4",
    },
    {
      category: "Replacement",
      instruction: "Replace the smooth skin tone with a weathered, sun-tanned complexion.",
      sourceVideo: "./assets/videos/cases/all6/all6-14-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-14-edited.png",
      video: "./assets/videos/cases/all6/all6-14.mp4",
    },
    {
      category: "Replacement",
      instruction: "Strumming an electric guitar",
      sourceVideo: "./assets/videos/cases/all6/all6-15-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-15-edited.png",
      video: "./assets/videos/cases/all6/all6-15.mp4",
    },
    {
      category: "Replacement",
      instruction: "Replace the pink fruit on top of the cake with sliced fresh strawberries.",
      sourceVideo: "./assets/videos/cases/all6/all6-16-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-16-edited.png",
      video: "./assets/videos/cases/all6/all6-16.mp4",
    },
    {
      category: "Texture",
      instruction: "Replace the dark wooden stock and forend with a lighter, polished wood finish.",
      sourceVideo: "./assets/videos/cases/all6/all6-22-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-22-edited.png",
      video: "./assets/videos/cases/all6/all6-22.mp4",
    },
    {
      category: "Texture",
      instruction: "Replace the yellow paint with a metallic blue finish.",
      sourceVideo: "./assets/videos/cases/all6/all6-23-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-23-edited.png",
      video: "./assets/videos/cases/all6/all6-23.mp4",
    },
    {
      category: "Texture",
      instruction: "Replace her dark hair with a lighter, sandy brown color.",
      sourceVideo: "./assets/videos/cases/all6/all6-24-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-24-edited.png",
      video: "./assets/videos/cases/all6/all6-24.mp4",
    },
    {
      category: "Shape/Style",
      instruction: "Replace the light pink roof with a dark red tiled roof.",
      sourceVideo: "./assets/videos/cases/all6/all6-26-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-26-edited.png",
      video: "./assets/videos/cases/all6/all6-26.mp4",
    },
    {
      category: "Shape/Style",
      instruction: "Replace the light brown wooden blocks with polished dark metal components.",
      sourceVideo: "./assets/videos/cases/all6/all6-28-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-28-edited.png",
      video: "./assets/videos/cases/all6/all6-28.mp4",
    },
    {
      category: "Shape/Style",
      instruction: "Replace the white lid with a black dome lid.",
      sourceVideo: "./assets/videos/cases/all6/all6-30-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-30-edited.png",
      video: "./assets/videos/cases/all6/all6-30.mp4",
    },
    {
      category: "Global",
      instruction: "Replace the smooth gray body material with a dark, textured scale armor plating.",
      sourceVideo: "./assets/videos/cases/all6/all6-32-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-32-edited.png",
      video: "./assets/videos/cases/all6/all6-32.mp4",
    },
    {
      category: "Global",
      instruction: "Gripping an ancient wooden staff",
      sourceVideo: "./assets/videos/cases/all6/all6-34-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-34-edited.png",
      video: "./assets/videos/cases/all6/all6-34.mp4",
    },
    {
      category: "Global",
      instruction: "Holding a camera ready to take a photo",
      sourceVideo: "./assets/videos/cases/all6/all6-35-source.mp4",
      editedImg: "./assets/images/cases/all6/all6-35-edited.png",
      video: "./assets/videos/cases/all6/all6-35.mp4",
    },
  ];


  // =====================================================================
  //  COMPARISON DATA — Baseline comparison cases
  // =====================================================================
  const comparisonData = [
    {
      instruction: 'Replace the irregular base with a solid, hexagonal platform.',
      sourceVideo: './assets/videos/compare/exp01-source.mp4',
      editedImg: './assets/images/compare/exp01-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp01-ours.mp4',
        'EditP23': './assets/videos/compare/exp01-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp01-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp01-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp01-voxhammer.mp4',
      }
    },
    {
      instruction: 'Add a small, leafy green plant placed on the wooden slab.',
      sourceVideo: './assets/videos/compare/exp02-source.mp4',
      editedImg: './assets/images/compare/exp02-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp02-ours.mp4',
        'EditP23': './assets/videos/compare/exp02-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp02-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp02-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp02-voxhammer.mp4',
      }
    },
    {
      instruction: 'Replace the white lid with a black dome lid.',
      sourceVideo: './assets/videos/compare/exp03-source.mp4',
      editedImg: './assets/images/compare/exp03-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp03-ours.mp4',
        'EditP23': './assets/videos/compare/exp03-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp03-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp03-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp03-voxhammer.mp4',
      }
    },
    {
      instruction: 'Gripping an ancient wooden staff',
      sourceVideo: './assets/videos/compare/exp04-source.mp4',
      editedImg: './assets/images/compare/exp04-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp04-ours.mp4',
        'EditP23': './assets/videos/compare/exp04-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp04-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp04-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp04-voxhammer.mp4',
      }
    },
    {
      instruction: 'Raising a flare gun high',
      sourceVideo: './assets/videos/compare/exp06-source.mp4',
      editedImg: './assets/images/compare/exp06-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp06-ours.mp4',
        'EditP23': './assets/videos/compare/exp06-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp06-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp06-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp06-voxhammer.mp4',
      }
    },
    {
      instruction: 'Replace the green hooded cloak with a dark grey trench coat.',
      sourceVideo: './assets/videos/compare/exp07-source.mp4',
      editedImg: './assets/images/compare/exp07-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp07-ours.mp4',
        'EditP23': './assets/videos/compare/exp07-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp07-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp07-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp07-voxhammer.mp4',
      }
    },
    {
      instruction: 'Playing a bamboo flute',
      sourceVideo: './assets/videos/compare/exp08-source.mp4',
      editedImg: './assets/images/compare/exp08-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp08-ours.mp4',
        'EditP23': './assets/videos/compare/exp08-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp08-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp08-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp08-voxhammer.mp4',
      }
    },
    {
      instruction: 'Add a small, golden sphere on the center of the glass disc.',
      sourceVideo: './assets/videos/compare/exp10-source.mp4',
      editedImg: './assets/images/compare/exp10-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp10-ours.mp4',
        'EditP23': './assets/videos/compare/exp10-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp10-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp10-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp10-voxhammer.mp4',
      }
    },
    {
      instruction: 'Waving a conductor\'s baton',
      sourceVideo: './assets/videos/compare/exp11-source.mp4',
      editedImg: './assets/images/compare/exp11-edited.png',
      methods: {
        'Ours': './assets/videos/compare/exp11-ours.mp4',
        'EditP23': './assets/videos/compare/exp11-editp23.mp4',
        'Instant3DiT': './assets/videos/compare/exp11-instant3dit.mp4',
        '3DEditFormer': './assets/videos/compare/exp11-3deditformer.mp4',
        'VoxHammer': './assets/videos/compare/exp11-voxhammer.mp4',
      }
    },
  ];


  // =====================================================================
  //  ABLATION DATA — Ablation visual comparison
  // =====================================================================
  const ablationData = [
    {
      instruction: 'Replace the smooth pale skin with iridescent, shifting scales.',
      sourceVideo: './assets/videos/ablation/ab01/source.mp4',
      editedImg: './assets/images/ablation/ab01-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab01/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab01/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab01/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab01/unilat3d.mp4',
      }
    },
    {
      instruction: 'Holding a camera ready to take a photo',
      sourceVideo: './assets/videos/ablation/ab02/source.mp4',
      editedImg: './assets/images/ablation/ab02-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab02/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab02/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab02/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab02/unilat3d.mp4',
      }
    },
    {
      instruction: 'Raising a flare gun high',
      sourceVideo: './assets/videos/ablation/ab03/source.mp4',
      editedImg: './assets/images/ablation/ab03-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab03/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab03/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab03/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab03/unilat3d.mp4',
      }
    },
    {
      instruction: 'Waving a conductor\'s baton',
      sourceVideo: './assets/videos/ablation/ab04/source.mp4',
      editedImg: './assets/images/ablation/ab04-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab04/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab04/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab04/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab04/unilat3d.mp4',
      }
    },
    {
      instruction: 'Waving a conductor\'s baton',
      sourceVideo: './assets/videos/ablation/ab05/source.mp4',
      editedImg: './assets/images/ablation/ab05-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab05/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab05/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab05/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab05/unilat3d.mp4',
      }
    },
    {
      instruction: 'Replace the dark wooden stock and forend with a lighter, polished wood finish.',
      sourceVideo: './assets/videos/ablation/ab06/source.mp4',
      editedImg: './assets/images/ablation/ab06-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab06/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab06/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab06/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab06/unilat3d.mp4',
      }
    },
    {
      instruction: 'Replace the checkered pattern on the fabric with a paisley pattern.',
      sourceVideo: './assets/videos/ablation/ab07/source.mp4',
      editedImg: './assets/images/ablation/ab07-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab07/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab07/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab07/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab07/unilat3d.mp4',
      }
    },
    {
      instruction: 'Replace the blue arm cuff on the giant\'s right arm with a silver spiked bracer.',
      sourceVideo: './assets/videos/ablation/ab08/source.mp4',
      editedImg: './assets/images/ablation/ab08-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab08/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab08/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab08/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab08/unilat3d.mp4',
      }
    },
    {
      instruction: 'Replace the military green color of the backpack with a camouflage pattern.',
      sourceVideo: './assets/videos/ablation/ab09/source.mp4',
      editedImg: './assets/images/ablation/ab09-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab09/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab09/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab09/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab09/unilat3d.mp4',
      }
    },
    {
      instruction: 'Replace her orange hair with black hair.',
      sourceVideo: './assets/videos/ablation/ab10/source.mp4',
      editedImg: './assets/images/ablation/ab10-edited.png',
      variants: {
        'Full (Ours)': './assets/videos/ablation/ab10/full.mp4',
        'w/o VLM': './assets/videos/ablation/ab10/w_o_vlm.mp4',
        'w/o VLM & DMD': './assets/videos/ablation/ab10/w_o_vlm_dmd.mp4',
        'UniLat3D': './assets/videos/ablation/ab10/unilat3d.mp4',
      }
    },
  ];


  // =====================================================================
  //  RENDER: Gallery with filter tabs
  // =====================================================================
  const uniqueCategories = [...new Set(galleryData.map(d => d.category))];
  const categories = uniqueCategories.includes('Global')
    ? ['Global', ...uniqueCategories.filter(c => c !== 'Global')]
    : uniqueCategories;
  const filterContainer = document.getElementById('gallery-filters');
  const gridContainer = document.getElementById('gallery-grid');

  if (filterContainer && gridContainer) {
    // Render filter buttons
    categories.forEach((cat, i) => {
      const btn = document.createElement('span');
      btn.className = 'tag-btn' + (i === 0 ? ' active' : '');
      btn.textContent = cat;
      btn.addEventListener('click', () => {
        filterContainer.querySelectorAll('.tag-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderGallery(cat);
      });
      filterContainer.appendChild(btn);
    });

    function renderGallery(activeCategory) {
      const filtered = galleryData.filter(d => d.category === activeCategory);

      gridContainer.innerHTML = filtered.map(item => `
        <div class="column is-one-third-desktop is-half-tablet">
          <div class="box gallery-card">
            <div class="card-content" style="padding:0.6rem 1rem 0.4rem;">
              <span class="instruction-tag">${item.category}</span>
              <p class="instruction-text">${item.instruction}</p>
            </div>
            <div style="display:flex;gap:6px;padding:0 0.5rem 0.5rem;">
              <div style="flex:0 0 31%;display:flex;flex-direction:column;gap:4px;">
                <div>
                  <p class="method-label">Source</p>
                  <div class="video-wrapper" style="padding-top:100%;">
                    <video autoplay loop muted playsinline>
                      <source src="${item.sourceVideo}" type="video/mp4">
                    </video>
                  </div>
                </div>
                <div>
                  <p class="method-label">Edited Image</p>
                  <div class="video-wrapper" style="padding-top:100%;"><img src="${item.editedImg}" alt="Edited"></div>
                </div>
              </div>
              <div style="flex:0 0 69%;">
                <p class="method-label ours-label">Ours (360°)</p>
                <div class="video-wrapper" style="padding-top:100%;">
                  <video autoplay loop muted playsinline>
                    <source src="${item.video}" type="video/mp4">
                  </video>
                </div>
              </div>
            </div>
          </div>
        </div>
      `).join('');
    }

    renderGallery(categories[0]);
  }


  // =====================================================================
  //  RENDER: Baseline comparisons
  // =====================================================================
  const cmpContainer = document.getElementById('comparison-container');
  if (cmpContainer) {
    cmpContainer.innerHTML = comparisonData.map(cmp => {
      const methodNames = Object.keys(cmp.methods);
      // Total columns = Source + Edited + methods count
      const totalCols = 2 + methodNames.length;
      // Use CSS flex with equal widths
      const colStyle = `flex:0 0 calc(${(100/totalCols).toFixed(2)}% - 8px);max-width:calc(${(100/totalCols).toFixed(2)}% - 8px);padding:4px;`;

      const videoCols = methodNames.map(name => `
        <div style="${colStyle}">
          <p class="method-label ${name === 'Ours' ? 'ours-label' : ''}">${name}</p>
          <div class="video-wrapper">
            <video autoplay loop muted playsinline>
              <source src="${cmp.methods[name]}" type="video/mp4">
            </video>
          </div>
        </div>
      `).join('');

      return `
        <div class="compare-block">
          <div class="compare-instruction">
            <strong>Instruction:</strong> ${cmp.instruction}
          </div>
          <div style="display:flex;flex-wrap:nowrap;gap:8px;align-items:flex-start;">
            <div style="${colStyle}">
              <p class="method-label">Source</p>
              <div class="video-wrapper">
                <video autoplay loop muted playsinline>
                  <source src="${cmp.sourceVideo}" type="video/mp4">
                </video>
              </div>
            </div>
            <div style="${colStyle}">
              <p class="method-label">Edited Image</p>
              <div class="video-wrapper"><img src="${cmp.editedImg}" alt="Edited"></div>
            </div>
            ${videoCols}
          </div>
        </div>
      `;
    }).join('');
  }


  // =====================================================================
  //  RENDER: Ablation
  // =====================================================================
  const ablContainer = document.getElementById('ablation-container');
  if (ablContainer) {
    ablContainer.innerHTML = ablationData.map(abl => {
      const variantNames = Object.keys(abl.variants);
      const totalCols = 2 + variantNames.length;
      const colStyle = `flex:0 0 calc(${(100/totalCols).toFixed(2)}% - 8px);max-width:calc(${(100/totalCols).toFixed(2)}% - 8px);padding:4px;`;

      const cols = variantNames.map(name => `
        <div style="${colStyle}">
          <p class="ablation-label" style="${name.includes('Ours') ? 'color:#c0392b;font-weight:700;' : ''}">${name}</p>
          <div class="video-wrapper">
            <video autoplay loop muted playsinline>
              <source src="${abl.variants[name]}" type="video/mp4">
            </video>
          </div>
        </div>
      `).join('');

      return `
        <div class="compare-block">
          <div class="compare-instruction">
            <strong>Instruction:</strong> ${abl.instruction}
          </div>
          <div style="display:flex;flex-wrap:nowrap;gap:8px;align-items:flex-start;">
            <div style="${colStyle}">
              <p class="ablation-label">Source</p>
              <div class="video-wrapper">
                <video autoplay loop muted playsinline>
                  <source src="${abl.sourceVideo}" type="video/mp4">
                </video>
              </div>
            </div>
            <div style="${colStyle}">
              <p class="ablation-label">Edited Image</p>
              <div class="video-wrapper"><img src="${abl.editedImg}" alt="Edited"></div>
            </div>
            ${cols}
          </div>
        </div>
      `;
    }).join('');
  }

})
