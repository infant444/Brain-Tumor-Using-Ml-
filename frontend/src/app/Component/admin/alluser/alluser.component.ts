import { Component ,OnInit} from '@angular/core';
import { User } from '../../../model/user';
import { AdminService } from '../../../Services/admin/admin.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-alluser',
  templateUrl: './alluser.component.html',
  styleUrl: './alluser.component.css'
})
export class AlluserComponent implements OnInit {
  users?:User[];
  profile="../../../../assets/images/profile.png";
  constructor(private adminService:AdminService,private toasterservice:ToastrService){}
  ngOnInit(): void {
    this.adminService.getAllUser().subscribe((users)=>{
      this.users=users
    })
  }

  del(id:string) {
    this.adminService.deleteuser(id).subscribe((det)=>{
      this.toasterservice.success("Successfully deleted")
      this.adminService.getAllUser().subscribe((users)=>{
        this.users=users
      })
    })
  }
}
