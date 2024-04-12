import { HttpEvent, HttpEventType, HttpHandler, HttpInterceptor, HttpInterceptorFn, HttpRequest } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, tap } from 'rxjs';
import { LoadingService } from '../../Services/loading/loading.service';

var pendingRequest=0;
@Injectable()
export class loadingInterceptor implements HttpInterceptor{

  constructor(private loading:LoadingService){}
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    this.loading.showLoading()
    pendingRequest+=1;
    return next.handle(req).pipe(
      tap({
        next:(event)=>{
          if(event.type==HttpEventType.Response){
            this.handleHideloading()
          }
        },
        error:(_)=>{
          this.handleHideloading()
        }
      })
    )
  }
  handleHideloading(){
    pendingRequest-=1;
    if(pendingRequest===0){
      this.loading.hideLoading();
    }
  }

}
