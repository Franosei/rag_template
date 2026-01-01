import logging
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import pytesseract
import hashlib
from io import BytesIO

from src.ingestion.models import (
    Chunk, Modality, ExtractionMethod, ConfidenceLevel,
    ImageMetadata
)
from src.utils.ids import generate_chunk_id

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Extract and process standalone images.
    Produces OCR text + caption/description.
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        enable_caption: bool = True,
        min_confidence: float = 0.7,
        llm_client=None  # For captioning
    ):
        self.enable_ocr = enable_ocr
        self.enable_caption = enable_caption
        self.min_confidence = min_confidence
        self.llm_client = llm_client
    
    async def extract_from_file(
        self,
        image_path: Path,
        doc_id: str,
        folder_id: str
    ) -> List[Chunk]:
        """Extract from standalone image file"""
        
        try:
            img = Image.open(image_path)
            
            return await self._process_image(
                image=img,
                doc_id=doc_id,
                folder_id=folder_id,
                file_path=str(image_path),
                file_name=image_path.name,
                image_index=0
            )
        
        except Exception as e:
            logger.error(f"Failed to extract from {image_path}: {e}")
            return []
    
    async def _process_image(
        self,
        image: Image.Image,
        doc_id: str,
        folder_id: str,
        file_path: str,
        file_name: str,
        image_index: int,
        page_number: Optional[int] = None,
        figure_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Process a single image: OCR + caption.
        Returns 1-2 chunks depending on what extraction succeeds.
        """
        
        chunks = []
        
        # Get image properties
        width, height = image.size
        img_format = image.format or "unknown"
        
        # Save image bytes for hashing
        img_bytes = BytesIO()
        image.save(img_bytes, format=image.format or "PNG")
        img_bytes.seek(0)
        content_hash = hashlib.md5(img_bytes.read()).hexdigest()
        
        # 1. OCR extraction
        ocr_text = None
        ocr_confidence = ConfidenceLevel.UNKNOWN
        
        if self.enable_ocr:
            try:
                # Run OCR
                ocr_data = pytesseract.image_to_data(
                    image,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text and calculate confidence
                words = ocr_data['text']
                confidences = ocr_data['conf']
                
                # Filter out empty strings and low confidence
                valid_words = []
                valid_confidences = []
                
                for word, conf in zip(words, confidences):
                    if word.strip() and conf != -1:
                        valid_words.append(word)
                        valid_confidences.append(float(conf))
                
                if valid_words:
                    ocr_text = " ".join(valid_words)
                    avg_confidence = sum(valid_confidences) / len(valid_confidences) / 100.0
                    
                    if avg_confidence >= 0.9:
                        ocr_confidence = ConfidenceLevel.HIGH
                    elif avg_confidence >= self.min_confidence:
                        ocr_confidence = ConfidenceLevel.MEDIUM
                    else:
                        ocr_confidence = ConfidenceLevel.LOW
                    
                    logger.debug(
                        f"OCR extracted {len(valid_words)} words with {avg_confidence:.2f} confidence",
                        extra={"file": file_name}
                    )
                else:
                    logger.debug(f"No text detected in image: {file_name}")
            
            except Exception as e:
                logger.warning(f"OCR failed for {file_name}: {e}")
        
        # 2. Caption generation (if LLM available)
        caption = None
        caption_confidence = ConfidenceLevel.UNKNOWN
        
        if self.enable_caption and self.llm_client:
            try:
                caption = await self._generate_caption(image, file_name)
                caption_confidence = ConfidenceLevel.HIGH  # Trust LLM captions
                
                logger.debug(f"Generated caption for {file_name}")
            
            except Exception as e:
                logger.warning(f"Caption generation failed for {file_name}: {e}")
        
        # 3. Create chunks based on what succeeded
        
        base_metadata = ImageMetadata(
            width=width,
            height=height,
            format=img_format,
            figure_number=figure_number,
            image_index=image_index,
            ocr_text=ocr_text,
            caption=caption
        )
        
        # If we have OCR text, create an OCR chunk
        if ocr_text:
            chunk_id = generate_chunk_id(doc_id, image_index * 10)  # Offset for OCR
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                folder_id=folder_id,
                source_doc_id=doc_id,
                modality=Modality.IMAGE,
                content_text=ocr_text,
                file_path=file_path,
                file_name=file_name,
                page_number=page_number,
                chunk_index=image_index * 10,
                extraction_method=ExtractionMethod.OCR,
                confidence=ocr_confidence,
                image_metadata=base_metadata,
                content_hash=f"{content_hash}_ocr"
            ))
        
        # If we have a caption, create a caption chunk
        if caption:
            chunk_id = generate_chunk_id(doc_id, image_index * 10 + 1)  # Offset for caption
            
            # Combine OCR + caption for richer content
            combined_text = f"{caption}"
            if ocr_text and len(ocr_text.strip()) > 0:
                combined_text += f"\n\nExtracted text: {ocr_text}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                folder_id=folder_id,
                source_doc_id=doc_id,
                modality=Modality.IMAGE,
                content_text=combined_text,
                file_path=file_path,
                file_name=file_name,
                page_number=page_number,
                chunk_index=image_index * 10 + 1,
                extraction_method=ExtractionMethod.CAPTION,
                confidence=caption_confidence,
                image_metadata=base_metadata,
                content_hash=f"{content_hash}_caption"
            ))
        
        # If neither succeeded, create a placeholder chunk with metadata
        if not chunks:
            chunk_id = generate_chunk_id(doc_id, image_index * 10)
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                folder_id=folder_id,
                source_doc_id=doc_id,
                modality=Modality.IMAGE,
                content_text=f"Image {image_index} from {file_name} (no text extracted)",
                file_path=file_path,
                file_name=file_name,
                page_number=page_number,
                chunk_index=image_index * 10,
                extraction_method=ExtractionMethod.PDF_IMAGE_EXTRACT,
                confidence=ConfidenceLevel.LOW,
                image_metadata=base_metadata,
                content_hash=content_hash
            ))
        
        return chunks
    
    async def _generate_caption(self, image: Image.Image, file_name: str) -> str:
        """
        Generate semantic description of image using vision model.
        In v1, we use GPT-4 Vision or similar.
        """
        if not self.llm_client:
            return ""
        
        # Convert image to base64 for API
        import base64
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Call vision model (OpenAI API format)
        # Note: This requires a vision-capable model
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Or gpt-4o
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Describe this image concisely for document retrieval purposes. Focus on:
                                - What type of image is it (chart, diagram, screenshot, form, photo)?
                                - Key visual elements (labels, axes, entities, text blocks)
                                - Main content or message

                                Keep it factual and under 100 words."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            caption = response.choices[0].message.content.strip()
            return caption
        
        except Exception as e:
            logger.warning(f"Vision API call failed: {e}")
            return ""


class PDFImageExtractor:
    """Extract images from PDF documents"""
    
    def __init__(self, image_processor: ImageExtractor):
        self.image_processor = image_processor
    
    async def extract_images_from_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        folder_id: str,
        is_scanned: bool = False
    ) -> List[Chunk]:
        """
        Extract embedded images OR render pages if scanned.
        
        For scanned PDFs: treat each page as an image
        For regular PDFs: extract embedded figures
        """
        import fitz  # PyMuPDF
        
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            if is_scanned:
                # Treat each page as an image
                chunks = await self._extract_pages_as_images(
                    doc, pdf_path, doc_id, folder_id
                )
            else:
                # Extract embedded images
                chunks = await self._extract_embedded_images(
                    doc, pdf_path, doc_id, folder_id
                )
            
            doc.close()
        
        except Exception as e:
            logger.error(f"PDF image extraction failed for {pdf_path}: {e}")
        
        return chunks
    
    async def _extract_embedded_images(
        self,
        doc,
        pdf_path: Path,
        doc_id: str,
        folder_id: str
    ) -> List[Chunk]:
        """Extract images embedded in PDF"""
        chunks = []
        image_index = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_ref in image_list:
                try:
                    xref = img_ref[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(BytesIO(image_bytes))
                    
                    # Process image
                    image_chunks = await self.image_processor._process_image(
                        image=image,
                        doc_id=doc_id,
                        folder_id=folder_id,
                        file_path=str(pdf_path),
                        file_name=pdf_path.name,
                        image_index=image_index,
                        page_number=page_num + 1,
                        figure_number=image_index + 1
                    )
                    
                    chunks.extend(image_chunks)
                    image_index += 1
                
                except Exception as e:
                    logger.warning(f"Failed to extract image {image_index} from page {page_num + 1}: {e}")
        
        logger.info(f"Extracted {image_index} images from {pdf_path.name}")
        return chunks
    
    async def _extract_pages_as_images(
        self,
        doc,
        pdf_path: Path,
        doc_id: str,
        folder_id: str
    ) -> List[Chunk]:
        """Render scanned PDF pages as images"""
        chunks = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # Render page to image (300 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_bytes = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(BytesIO(img_bytes))
                
                # Process as image
                page_chunks = await self.image_processor._process_image(
                    image=image,
                    doc_id=doc_id,
                    folder_id=folder_id,
                    file_path=str(pdf_path),
                    file_name=pdf_path.name,
                    image_index=page_num,
                    page_number=page_num + 1
                )
                
                chunks.extend(page_chunks)
            
            except Exception as e:
                logger.warning(f"Failed to render page {page_num + 1} of {pdf_path.name}: {e}")
        
        logger.info(f"Rendered {len(doc)} pages from scanned PDF: {pdf_path.name}")
        return chunks


class DOCXImageExtractor:
    """Extract images from Word documents"""
    
    def __init__(self, image_processor: ImageExtractor):
        self.image_processor = image_processor
    
    async def extract_images_from_docx(
        self,
        docx_path: Path,
        doc_id: str,
        folder_id: str
    ) -> List[Chunk]:
        """Extract embedded images from DOCX"""
        from docx import Document
        
        chunks = []
        
        try:
            doc = Document(docx_path)
            image_index = 0
            
            # Iterate through relationships to find images
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_bytes = rel.target_part.blob
                        image = Image.open(BytesIO(image_bytes))
                        
                        # Process image
                        image_chunks = await self.image_processor._process_image(
                            image=image,
                            doc_id=doc_id,
                            folder_id=folder_id,
                            file_path=str(docx_path),
                            file_name=docx_path.name,
                            image_index=image_index,
                            figure_number=image_index + 1
                        )
                        
                        chunks.extend(image_chunks)
                        image_index += 1
                    
                    except Exception as e:
                        logger.warning(f"Failed to extract image {image_index} from {docx_path.name}: {e}")
            
            logger.info(f"Extracted {image_index} images from {docx_path.name}")
        
        except Exception as e:
            logger.error(f"DOCX image extraction failed for {docx_path}: {e}")
        
        return chunks