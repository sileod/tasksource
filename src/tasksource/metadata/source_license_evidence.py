"""Original dataset terms missed by Hub cards; manually checked 2026-10-09.

These are publisher statements, not inferred licenses from annotation software.
Keys are task families, shared by their visual classification/MC views.
"""

SOURCE_LICENSE_EVIDENCE = {
    'clevr': {'license': 'cc-by-4.0', 'scope': 'all dataset data',
              'url': 'https://cs.stanford.edu/people/jcjohns/clevr/'},
    'mapqa': {'license': 'cc-by-sa-4.0', 'scope': 'dataset repository (data and code)',
              'url': 'https://github.com/OSU-slatelab/MapQA/blob/main/license'},
    'tqa': {'license': 'cc-by-sa-4.0', 'scope': 'dataset distribution',
            'url': 'https://registry.opendata.aws/allenai-tqa/'},
}

# Scoped review: a permissive annotation/repository license cannot clear photos.
# ``license_use`` applies conservatively to the complete image + annotation row.
SOURCE_LICENSE_REVIEWS = {
    'aokvqa': {
        'license_use': 'unspecified', 'status': 'partial',
        'annotations': {'license': 'apache-2.0', 'scope': 'original dataset repository',
                        'url': 'https://github.com/allenai/aokvqa/blob/main/LICENSE'},
        'images': {'license': 'per-image COCO/Flickr terms', 'url': 'https://cocodataset.org/#termsofuse'},
        'unresolved': 'COCO image licenses and attribution have not been joined to the exported rows.',
    },
    'nlvr2': {
        'license_use': 'unspecified', 'status': 'partial',
        'annotations': {'license': 'cc-by-4.0', 'scope': 'sentences and binary labels',
                        'url': 'https://github.com/lil-lab/nlvr#licensing'},
        'images': {'license': 'not licensed by dataset authors',
                   'url': 'https://github.com/lil-lab/nlvr#licensing'},
        'unresolved': 'The authors explicitly do not license the photographs; original rights remain unresolved.',
    },
    'tallyqa': {
        'license_use': 'unspecified', 'status': 'partial',
        'annotations': {'license': 'apache-2.0', 'scope': 'original dataset repository; includes imported QAs',
                        'url': 'https://github.com/manoja328/TallyQA_dataset/blob/master/LICENSE'},
        'images': {'license': 'COCO per-image terms / Visual Genome cc-by-4.0',
                   'url': 'https://github.com/manoja328/TallyQA_dataset#download-images',
                   'additional_urls': ['https://cocodataset.org/#termsofuse',
                                       'https://homes.cs.washington.edu/~ranjay/visualgenome/about.html']},
        'unresolved': 'Image origins/licenses and imported QA terms have not been resolved per exported row.',
    },
    'vsr': {
        'license_use': 'unspecified', 'status': 'partial',
        'annotations': {'license': 'cc-by-4.0', 'scope': 'official annotation dataset card',
                        'url': 'https://huggingface.co/datasets/cambridgeltl/vsr_random/blob/main/README.md'},
        'images': {'license': 'per-image COCO/Flickr terms', 'url': 'https://cocodataset.org/#termsofuse'},
        'unresolved': 'COCO image licenses and attribution have not been joined to the exported rows.',
    },
    'figureqa': {
        'license_use': 'non-commercial', 'status': 'resolved',
        'annotations': {'license': 'Microsoft Research Open Data License', 'scope': 'dataset and accompanying content'},
        'images': {'license': 'Microsoft Research Open Data License'},
        'url': 'https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-sample-train-v1.tar.gz',
        'archive_member': 'sample_train1/MSR Open Data Research License.pdf',
        'redistribution': 'prohibited',
        'notes': 'Sections 1-3 restrict use to non-commercial research/testing and prohibit dataset redistribution/hosting. Generator MIT terms cover code only.',
    },
    'bapps': {
        'license_use': 'non-commercial', 'status': 'partial',
        'annotations': {'license': 'unspecified', 'scope': 'human perceptual judgments'},
        'images': {'license': 'Adobe / Adobe-MIT Research License (training images)',
                   'url': 'https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt',
                   'additional_urls': ['https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt']},
        'url': 'https://arxiv.org/abs/1801.03924',
        'notes': 'Original paper section 3 identifies MIT-Adobe FiveK training patches; image licenses require research use without commercial advantage and retention of notices.',
        'unresolved': 'Judgment-specific terms and licenses of the distinct validation image sources remain unresolved; family classification conservatively reflects restricted training images.',
    },
    'visual7w': {
        'license_use': 'unspecified', 'status': 'unresolved',
        'annotations': {'license': 'unspecified', 'url': 'https://ai.stanford.edu/~yukez/visual7w/'},
        'images': {'license': 'upstream Visual Genome / original image terms'},
        'unresolved': 'No explicit dataset grant found on the original project page or toolkit dataset documentation; toolkit MIT license is not a photo license.',
    },
    'intergps': {
        'license_use': 'unspecified', 'status': 'partial',
        'annotations': {'license': 'mit', 'scope': 'original data/code repository',
                        'url': 'https://github.com/lupantech/InterGPS/blob/main/LICENSE'},
        'images': {'license': 'unspecified third-party geometry diagrams'},
        'unresolved': 'Repository MIT terms do not establish underlying textbook/problem image rights.',
    },
    'spair71k': {
        'license_use': 'unspecified', 'status': 'partial',
        'annotations': {'license': 'inherited PASCAL-VOC/Flickr terms'},
        'images': {'license': 'inherited PASCAL-VOC/Flickr terms'},
        'url': 'https://cvlab.postech.ac.kr/research/SPair-71k/index.html',
        'unresolved': 'Official terms defer images and metadata to PASCAL/Flickr; individual image grants have not been joined.',
    },
}
