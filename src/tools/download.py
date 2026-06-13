import os
import time
import cloudscraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_pdf(url: str, output_path: str) -> bool:
    """
    Realiza o download de um arquivo PDF.
    Retorna booleano indicando o sucesso da operacao.
    """
    try:
        # Rate limit basal
        time.sleep(0.1) 
        
        # Utiliza scraper configurado para contornar protecoes anti-bot
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        
        response = scraper.get(url, timeout=30.0)
        
        if response.status_code == 200:
            # Validacao do magic number para garantir integridade do PDF
            if b"%PDF-" not in response.content[:1024]:
                logger.warning(f"Arquivo baixado falhou na verificacao de assinatura PDF: {url}")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Download concluido: {output_path}")
            return True
        else:
            logger.error(f"Requisicao falhou com HTTP {response.status_code} para a url {url}")
            return False
            
    except Exception as e:
        logger.error(f"Excecao durante o download de {url}: {str(e)}")
        return False